"""Registry for the Mean-ResNet downstream model."""

from downstream.mean_resnet.model import MeanResNet
from downstream.shared_model import build_model, build_scratch_model as _build_scratch_model

_MODEL_CLASSES = {MeanResNet.model_name: MeanResNet}
_LEGACY_POOLING_TO_MODEL = {MeanResNet.pooling_name: MeanResNet.model_name}


def available_models():
    return tuple(_MODEL_CLASSES)


def get_model_class(model_name):
    try:
        return _MODEL_CLASSES[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown downstream model {model_name!r}; choose from {available_models()}") from exc


def checkpoint_model_name(saved):
    model_name = saved.get("model_name")
    if model_name is not None:
        get_model_class(model_name)
        return model_name
    pooling = saved.get("pooling")
    try:
        return _LEGACY_POOLING_TO_MODEL[pooling]
    except KeyError as exc:
        raise ValueError("Checkpoint has no recognized downstream model identity") from exc


def build_from_spec(model_name, sr_root, model_spec, device):
    return build_model(get_model_class(model_name), sr_root, model_spec, device)


def build_scratch_model(model_name, sr_root, model_spec, device):
    return _build_scratch_model(get_model_class(model_name), sr_root, model_spec, device)
