"""Compatibility layer for the independent Mean-ResNet model."""

from downstream.mean_resnet.model import MeanResNet as E2EModel
from downstream.registry import build_from_spec as _build_from_spec
from downstream.shared_model import (_check_shape, _source_models, _temporary_bn_buffers,
                                     install_hat_checkpointing)


def build_from_spec(sr_root, model_spec, device):
    return _build_from_spec("mean_resnet", sr_root, model_spec, device)


__all__ = ("E2EModel", "build_from_spec", "_check_shape", "_source_models",
           "_temporary_bn_buffers", "install_hat_checkpointing")
