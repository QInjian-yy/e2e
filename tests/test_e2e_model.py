"""CPU checks of checkpoint mathematics, not an 8K CUDA/gsplat smoke test."""

import ast
import copy
import math
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from downstream.mean_resnet.model import MeanResNet as E2EModel
from downstream.shared_model import _temporary_bn_buffers, install_hat_checkpointing


def source_hat_class():
    """Load real local HAT bodies without importing gsplat or basicsr on CPU.

    Only external registry/tuple/init helpers and the single einops permutation
    are supplied here. HAB, OCAB, AttenBlocks, RHAG and HAT bodies stay unchanged.
    """
    default_root = (Path(__file__).resolve().parents[2] /
                    "infer/home/yusicheng/shendongxu/data/ContinuousSR")
    path = Path(os.environ.get("SR_ROOT", str(default_root))) / "models" / "hat.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]

    def rearrange(value, pattern, nc, ch, owh, oww):
        assert pattern == "b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch"
        batch, _, windows = value.shape
        return (value.reshape(batch, nc, ch, owh * oww, windows)
                .permute(1, 0, 4, 3, 2).reshape(nc, batch * windows, owh * oww, ch))

    namespace = {"__name__": "source_hat_cpu_test", "math": math, "torch": torch,
                 "nn": nn, "F": F, "checkpoint": torch.utils.checkpoint,
                 "to_2tuple": lambda value: (value, value) if isinstance(value, int) else value,
                 "trunc_normal_": nn.init.trunc_normal_, "rearrange": rearrange,
                 "register": lambda name: lambda cls: cls}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["HAT"]


def small_hat(size=8, depths=(2, 2)):
    return source_hat_class()(
        img_size=size, embed_dim=8, depths=depths, num_heads=(2,) * len(depths),
        window_size=4, compress_ratio=2, squeeze_factor=2, overlap_ratio=0.5,
        mlp_ratio=2, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.2,
        upscale=4, upsampler="pixelshuffle",
    )


class ToySR(nn.Module):
    """CPU stand-in for wrapper control flow only, not a Gaussian implementation."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.conv_first = nn.Conv2d(3, 64, 1)
        self.encoder.upsample = nn.Upsample(scale_factor=4, mode="nearest")
        self.ps = nn.PixelUnshuffle(2)
        self.color = nn.Conv2d(256, 3, 1)
        self.inp = self.feat = None
        self.query_batches = []
        self.query_input_means = []
        self.fail_on_call = None
        self.last_output = None

    def extract_features(self, lr):
        return self.encoder.conv_first(lr)

    def query_output(self, inp, scale):
        self.query_batches.append(inp.shape[0])
        self.query_input_means.append(inp.detach().mean().item())
        assert inp.shape[0] == self.feat.shape[0] == 1
        assert self.feat.dtype == torch.float32
        assert scale.item() == 32
        if len(self.query_batches) == self.fail_on_call:
            raise RuntimeError("fake Gaussian query failed")
        self.last_output = F.interpolate(
            self.color(self.feat), size=(inp.shape[-2] * 32, inp.shape[-1] * 32), mode="nearest"
        )
        return self.last_output


def toy_sr_wrapper():
    # Exercise the production forward_sr method without constructing CUDA gsplat.
    model = E2EModel.__new__(E2EModel)
    nn.Module.__init__(model)
    model.sr = ToySR()
    return model


def check_small_wrapper_shape(tensor, expected, name, log=False):
    # Test-only 1/128 spatial sizes: LR2 -> upsample8 -> unshuffle4 -> output64.
    # The production checker and its mandatory LR256/output8192 remain untouched.
    reduced = (*expected[:2], expected[2] // 128, expected[3] // 128)
    if tuple(tensor.shape) != reduced:
        raise AssertionError(f"{name}: expected test shape {reduced}, got {tuple(tensor.shape)}")


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_parameter_gradients_equal(self, reference, actual):
        reference_params = dict(reference.named_parameters())
        self.assertEqual(reference_params.keys(), dict(actual.named_parameters()).keys())
        for name, param in actual.named_parameters():
            expected = reference_params[name].grad
            if expected is None:
                self.assertIsNone(param.grad, name)
            else:
                self.assertIsNotNone(param.grad, name)
                torch.testing.assert_close(param.grad, expected, rtol=2e-5, atol=2e-6, msg=name)

    def test_real_hat_each_hab_ocab_matches_outputs_and_gradients(self):
        torch.manual_seed(4)
        reference = small_hat().train()
        actual = copy.deepcopy(reference)
        self.assertEqual(install_hat_checkpointing(actual), {"RHAG": 2, "HAB": 4, "OCAB": 2})
        lr = torch.randn(2, 3, 8, 8, requires_grad=True)
        lr_checkpointed = lr.detach().clone().requires_grad_(True)
        torch.manual_seed(9)
        expected = reference.extract_features(lr)
        expected.square().mean().backward()
        torch.manual_seed(9)
        output = actual.extract_features(lr_checkpointed)
        output.square().mean().backward()
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(lr_checkpointed.grad, lr.grad)
        self.assert_parameter_gradients_equal(reference, actual)

    def test_recompute_buffers_restore_after_exception_and_cumulative_bn(self):
        bn = nn.BatchNorm2d(3, momentum=None).train()
        bn(torch.randn(2, 3, 4, 4))
        buffers = dict(bn.named_buffers())
        values = {name: value.clone() for name, value in buffers.items()}
        with self.assertRaisesRegex(RuntimeError, "stop recomputation"):
            with _temporary_bn_buffers(bn):
                bn(torch.randn(2, 3, 4, 4))
                raise RuntimeError("stop recomputation")
        for name, value in bn.named_buffers():
            self.assertIs(value, buffers[name])
            torch.testing.assert_close(value, values[name])

    def test_sr_wrapper_microbatches_match_independent_outputs_and_gradients(self):
        # This validates CPU batching/autograd/cache behavior, not gsplat or 8K.
        for batch in (2, 4):
            with self.subTest(batch=batch):
                torch.manual_seed(61)
                actual = toy_sr_wrapper()
                reference = copy.deepcopy(actual)
                lr = (torch.arange(batch * 12).reshape(batch, 3, 2, 2).float() / 10).requires_grad_()
                reference_lr = lr.detach().clone().requires_grad_()
                with patch("downstream.shared_model._check_shape", side_effect=check_small_wrapper_shape):
                    output = actual.forward_sr(lr)
                    expected = torch.cat([reference.forward_sr(sample)
                                          for sample in reference_lr.split(1)], dim=0)
                output.square().mean().backward()
                expected.square().mean().backward()
                torch.testing.assert_close(output, expected)
                torch.testing.assert_close(lr.grad, reference_lr.grad)
                self.assert_parameter_gradients_equal(reference, actual)
                self.assertEqual(actual.sr.query_batches, [1] * batch)
                self.assertEqual(actual.sr.query_input_means, reference.sr.query_input_means)
                self.assertIsNone(actual.sr.inp)
                self.assertIsNone(actual.sr.feat)
                self.assertGreater(actual.sr.encoder.conv_first.weight.grad.norm().item(), 0)

    def test_sr_wrapper_clears_cache_when_query_raises(self):
        model = toy_sr_wrapper()
        model.sr.fail_on_call = 2
        with patch("downstream.shared_model._check_shape", side_effect=check_small_wrapper_shape):
            with self.assertRaisesRegex(RuntimeError, "fake Gaussian query failed"):
                model.forward_sr(torch.randn(4, 3, 2, 2))
        self.assertEqual(model.sr.query_batches, [1, 1])
        self.assertIsNone(model.sr.inp)
        self.assertIsNone(model.sr.feat)

    def test_sr_wrapper_single_sample_has_no_extra_cat_and_casts_query_to_fp32(self):
        model = toy_sr_wrapper()
        # Simulate BF16 HAT/upsample output without requiring CUDA autocast.
        hook = model.sr.ps.register_forward_hook(lambda module, inputs, value: value.bfloat16())
        with patch("downstream.shared_model._check_shape", side_effect=check_small_wrapper_shape):
            with patch("downstream.shared_model.torch.cat", side_effect=AssertionError("unnecessary 8K cat")):
                output = model.forward_sr(torch.randn(1, 3, 2, 2))
        hook.remove()
        self.assertIs(output, model.sr.last_output)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(model.sr.query_batches, [1])
        output.mean().backward()
        self.assertGreater(model.sr.encoder.conv_first.weight.grad.norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
