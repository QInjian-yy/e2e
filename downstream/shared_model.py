"""Shared HAT/Gaussian and ResNet18 region-encoding infrastructure."""

import copy
import importlib
import inspect
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet18

_CKPT_PARAMS = inspect.signature(checkpoint).parameters


def _replace_bn_with_gn(module, num_groups=32):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            channels = child.num_features
            groups = min(num_groups, channels)
            while channels % groups:
                groups -= 1
            setattr(module, name, nn.GroupNorm(groups, channels))
        else:
            _replace_bn_with_gn(child, num_groups)


def _build_region_encoder():
    encoder = resnet18(weights=None)
    encoder.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
    encoder.fc = nn.Identity()
    _replace_bn_with_gn(encoder)
    return encoder


def _run_checkpoint(function, *args):
    kwargs = {}
    if "use_reentrant" in _CKPT_PARAMS:
        kwargs["use_reentrant"] = False
    return checkpoint(function, *args, **kwargs)


def _attention_forward(self, x, x_size, params):
    """The original AttenBlocks order, checkpointing each HAB and OCAB."""
    use_checkpoint = self.training and torch.is_grad_enabled()
    for block in self.blocks:
        args = (x, x_size, params["rpi_sa"], params["attn_mask"])
        x = _run_checkpoint(block, *args) if use_checkpoint else block(*args)
    args = (x, x_size, params["rpi_oca"])
    x = (_run_checkpoint(self.overlap_attn, *args)
         if use_checkpoint else self.overlap_attn(*args))
    if self.downsample is not None:
        x = self.downsample(x)
    return x


def install_hat_checkpointing(encoder):
    """Patch only this E2E model's instances; leave the source files untouched."""
    groups = [module for module in encoder.modules()
              if module.__class__.__name__ == "AttenBlocks"]
    if not groups:
        raise ValueError("The loaded HAT has no AttenBlocks to checkpoint")
    for group in groups:
        group.forward = MethodType(_attention_forward, group)
        group.use_checkpoint = True
    return {"RHAG": len(groups), "HAB": sum(len(group.blocks) for group in groups),
            "OCAB": len(groups)}


@contextmanager
def _temporary_bn_buffers(module):
    """Let replay use training-mode BN without updating real buffers twice."""
    originals = []
    try:
        for child in module.modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm) and child.training:
                for name in ("running_mean", "running_var", "num_batches_tracked"):
                    value = getattr(child, name)
                    if value is not None:
                        originals.append((child, name, value))
                        setattr(child, name, value.clone())
        yield
    finally:
        for child, name, value in originals:
            setattr(child, name, value)


def _check_shape(tensor, expected, name, log=False):
    if log:
        print(f"{name}: {list(tensor.shape)}; dtype={tensor.dtype}", flush=True)
    if tuple(tensor.shape) != tuple(expected):
        raise ValueError(f"{name}: expected {list(expected)}, got {list(tensor.shape)}")


class SharedE2EModel(nn.Module):
    """Infrastructure shared by independent downstream model implementations."""

    model_name = None
    pooling_name = None
    downstream_description = None

    def __init__(self, sr, model_spec=None):
        super().__init__()
        self.sr = sr  # The only registered HAT is sr.encoder.
        if sr.encoder.upscale != 4 or sr.encoder.upsampler != "pixelshuffle":
            raise ValueError("The existing Gaussian head requires HAT x4 pixelshuffle")
        self.feature_channels = sr.encoder.conv_before_upsample[0].out_channels
        if self.feature_channels != 64:
            raise ValueError(f"Expected 64-channel HAT feature; got {self.feature_channels}")
        self.checkpoint_counts = install_hat_checkpointing(sr.encoder)
        self.region_encoder = resnet18(weights=None)
        self.region_encoder.conv1 = nn.Conv2d(64, 64, 7, stride=2, padding=3, bias=False)
        self.region_encoder.fc = nn.Identity()
        self.model_spec = copy.deepcopy(model_spec)
        self.requires_grad_(True)

    def encode_regions(self, lr, preserve_bn_stats=False, log_shapes=False):
        """Encode one region micro-batch; replay alone uses temporary BN buffers."""
        _check_shape(lr, (lr.shape[0], 3, 256, 256), "classification LR", log_shapes)
        bn_ctx = (_temporary_bn_buffers(self.region_encoder)
                  if preserve_bn_stats and self.training else nullcontext())
        with bn_ctx:
            with torch.autocast(device_type=lr.device.type, dtype=torch.bfloat16,
                                enabled=lr.is_cuda):
                feature = self.sr.extract_features(lr)
                _check_shape(feature, (lr.shape[0], 64, 256, 256), "HAT feature")
                embedding = self.region_encoder(feature)
        _check_shape(embedding, (lr.shape[0], 512), "region embedding", log_shapes)
        if log_shapes:
            print(f"HAT feature: {[lr.shape[0], 64, 256, 256]} (checked inside encoder)", flush=True)
        return embedding

    def aggregate_embeddings(self, embeddings):
        raise NotImplementedError

    def classify(self, wsi_embedding):
        raise NotImplementedError

    def forward_embeddings(self, embeddings):
        return self.classify(self.aggregate_embeddings(embeddings))

    def downstream_gradient_groups(self):
        raise NotImplementedError

    def forward_sr(self, lr, log_shapes=False):
        """Generate complete 8K outputs for this micro-batch only."""
        batch = lr.shape[0]
        _check_shape(lr, (batch, 3, 256, 256), "SR LR", log_shapes)
        with torch.autocast(device_type=lr.device.type, dtype=torch.bfloat16, enabled=lr.is_cuda):
            feature = self.sr.extract_features(lr)
            _check_shape(feature, (batch, 64, 256, 256), "HAT feature", log_shapes)
            upsampled = self.sr.encoder.upsample(feature)
            _check_shape(upsampled, (batch, 64, 1024, 1024), "HAT x4 upsample", log_shapes)
            gaussian_feature = self.sr.ps(upsampled)
            _check_shape(gaussian_feature, (batch, 256, 512, 512),
                         "PixelUnshuffle / Gaussian input", log_shapes)
            del feature, upsampled
        gaussian_feature = gaussian_feature.float()
        outputs = []
        try:
            with torch.autocast(device_type=lr.device.type, enabled=False):
                scale = lr.new_tensor([32.0], dtype=torch.float32)
                for index in range(batch):
                    self.sr.inp = lr[index:index + 1]
                    self.sr.feat = gaussian_feature[index:index + 1]
                    output = self.sr.query_output(self.sr.inp, scale)
                    _check_shape(output, (1, 3, 8192, 8192), "Gaussian sample output", log_shapes)
                    outputs.append(output)
        finally:
            self.clear_sr_cache()
        result = outputs[0] if batch == 1 else torch.cat(outputs, dim=0)
        _check_shape(result, (batch, 3, 8192, 8192), "SR output", log_shapes)
        return result

    def clear_sr_cache(self):
        self.sr.inp = None
        self.sr.feat = None


def _source_models(sr_root):
    root = Path(sr_root).resolve()
    if not (root / "models" / "__init__.py").is_file() or not (root / "utils.py").is_file():
        raise FileNotFoundError(f"--sr-root must contain the original models package and utils.py: {root}")
    existing = sys.modules.get("models")
    if existing is not None and Path(existing.__file__).resolve().parent != root / "models":
        raise RuntimeError(f"A different 'models' package is already imported: {existing.__file__}")
    sys.path.insert(0, str(root))
    models = importlib.import_module("models")
    print(f"ContinuousSR source: {models.__file__}", flush=True)
    return models


def build_model(model_class, sr_root, spec, device, downstream_args=None):
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("The original Gaussian constructor and gsplat require CUDA")
    torch.cuda.set_device(device)
    if spec["name"] != "continuous-gaussian" or spec["args"]["encoder_spec"]["name"] != "hat":
        raise ValueError("Expected original continuous-gaussian with a HAT encoder")
    models = _source_models(sr_root)
    sr = models.make(spec, load_sd=False)
    architecture = {"name": spec["name"], "args": copy.deepcopy(spec["args"])}
    if downstream_args is None:
        model = model_class(sr, architecture).to(device)
    else:
        model = model_class(sr, architecture, downstream_args=downstream_args).to(device)
    print(f"Downstream model: {model.model_name}", flush=True)
    print(f"HAT config: {architecture['args']['encoder_spec']['args']}", flush=True)
    print(f"HAT checkpoints: {model.checkpoint_counts}; every HAB / OCAB, non-reentrant", flush=True)
    memory = ("gradient caching replay" if getattr(model, "classification_gradpool", True)
              else "full classification graph")
    print(f"Classification memory: {memory}; no outer HAT+ResNet18 checkpoint", flush=True)
    print("Replay protection: per-micro-batch RNG restore + temporary ResNet18 BN buffers", flush=True)
    print(model.downstream_description, flush=True)
    print("Gaussian query runs separately per sample; Gaussian FP32, HAT/ResNet18 BF16 autocast", flush=True)
    return model


def build_scratch_model(model_class, sr_root, model_spec, device, downstream_args=None):
    """Build HAT, Gaussian, ResNet18 and downstream parameters from random initialization."""
    model = build_model(model_class, sr_root, model_spec, device, downstream_args)
    print("SR initialization: random weights from --sr-config (no SR checkpoint loaded)", flush=True)
    return model
