"""CPU interface checks for the Mean-ResNet downstream model."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from downstream.mean_resnet.model import MeanResNet
from downstream.registry import (available_models, build_scratch_model,
                                 checkpoint_model_name, get_model_class)
from downstream.shared_model import SharedE2EModel


class DownstreamModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_registry_contains_mean_and_wikg(self):
        self.assertEqual(available_models(), ("mean_resnet", "resnet_wikg_abmil"))
        self.assertIs(get_model_class("mean_resnet"), MeanResNet)
        self.assertEqual(checkpoint_model_name({"pooling": "mean"}), "mean_resnet")
        self.assertEqual(checkpoint_model_name({"model_name": "mean_resnet"}), "mean_resnet")

    def test_scratch_builder_uses_mean_resnet(self):
        sentinel = object()
        with patch("downstream.registry._build_scratch_model", return_value=sentinel) as build:
            result = build_scratch_model("mean_resnet", "sr-root", {"name": "spec"}, "cuda:0")
        self.assertIs(result, sentinel)
        build.assert_called_once_with(MeanResNet, "sr-root", {"name": "spec"}, "cuda:0")

    def build_without_backbone(self, seed=19):
        def initialize(module, sr, model_spec=None):
            nn.Module.__init__(module)

        torch.manual_seed(seed)
        with patch.object(SharedE2EModel, "__init__", initialize):
            return MeanResNet(object())

    def test_mean_embedding_interface(self):
        model = self.build_without_backbone()
        embeddings = torch.randn(7, 512, requires_grad=True)
        pooled = model.aggregate_embeddings(embeddings)
        logits = model.forward_embeddings(embeddings)
        self.assertEqual(pooled.shape, (1, 512))
        self.assertEqual(logits.shape, (1, 2))
        logits.square().mean().backward()
        self.assertIsNotNone(embeddings.grad)
        self.assertTrue(torch.isfinite(embeddings.grad).all())

    def test_real_resnet18_minimal_forward_backward(self):
        class TinyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.upscale = 4
                self.upsampler = "pixelshuffle"
                self.conv_first = nn.Conv2d(3, 64, 1)
                self.conv_before_upsample = nn.Sequential(nn.Conv2d(64, 64, 1))

            def forward(self, inputs):
                return self.conv_before_upsample(self.conv_first(inputs))

        class TinySR(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = TinyEncoder()

            def extract_features(self, inputs):
                return self.encoder(inputs)

        with patch("downstream.shared_model.install_hat_checkpointing",
                   return_value={"RHAG": 1, "HAB": 1, "OCAB": 1}), \
                patch("downstream.shared_model._check_shape"):
            model = MeanResNet(TinySR()).train()
            embeddings = model.encode_regions(torch.randn(2, 3, 8, 8))
            logits = model.forward_embeddings(embeddings)
            loss = F.cross_entropy(logits, torch.tensor([1]))
            loss.backward()
            self.assertEqual(embeddings.shape, (2, 512))
            self.assertEqual(logits.shape, (1, 2))
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNotNone(model.sr.encoder.conv_first.weight.grad)
            self.assertIsNotNone(model.region_encoder.conv1.weight.grad)
            self.assertIsNotNone(model.classifier.weight.grad)


if __name__ == "__main__":
    unittest.main()
