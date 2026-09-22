"""Synthetic checks; real HAT bodies + ResNet18, no CUDA Gaussian/8K rendering."""

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from downstream.registry import build_from_spec, checkpoint_model_name, get_model_class
from downstream.resnet_wikg_abmil.model import GatedAttentionMIL, ResNetWiKGABMIL, WiKGCommunication
from downstream.shared_model import SharedE2EModel
from training import engine, train
from test_e2e_model import small_hat


ARGS = dict(wikg_topk=6, wikg_dropout=0.3, classification_gradpool=True, debug_shapes=False)


class SmallSR(nn.Module):
    def __init__(self, size=64):
        super().__init__()
        self.encoder = small_hat(size=size, depths=(1,))
        self.head = nn.Conv2d(64, 3, 1)
        self.inp = self.feat = None

    def extract_features(self, x):
        return self.encoder.extract_features(x)


class SmallSRWiKG(ResNetWiKGABMIL):
    def forward_sr(self, lr, log_shapes=False):
        # Only the SR branch is a stand-in; classification modules are real.
        return self.sr.head(self.sr.extract_features(lr))


def shape_check_small(tensor, expected, name, log=False):
    if name in ("classification LR", "HAT feature"):
        expected = (*expected[:2], tensor.shape[-2], tensor.shape[-1])
    if tuple(tensor.shape) != tuple(expected):
        raise AssertionError((name, tuple(tensor.shape), expected))


def sample_for(n):
    return dict(slide_id="synthetic_wikg", label=1, n_regions=n,
                lr_paths=list(range(n)), hr_paths=list(range(n)))


def equivalence_probe(dropout, device="cpu", size=64, n=3, micro_batch=2):
    """Same weights/data/RNG/chunks; return measured differences, not just success."""
    device = torch.device(device)
    torch.manual_seed(193)
    args = dict(ARGS, wikg_dropout=dropout)
    full = SmallSRWiKG(SmallSR(size), downstream_args=args).to(device).train()
    full.classification_gradpool = False
    cached = copy.deepcopy(full)
    cached.classification_gradpool = True
    images = torch.randn(n, 3, size, size)
    target = torch.tensor([1], device=device)
    results, logits, rngs, diagnostics = [], [], [], []
    for model, helper in ((full, engine.full_graph_classification_loss_and_backward),
                          (cached, engine.classification_loss_and_backward)):
        torch.manual_seed(417)
        diag, captured, bags = {}, [], []
        handle = model.classifier.register_forward_hook(
            lambda module, inputs, output: captured.append(output.detach().float().cpu()))
        bag_handle = model.wikg.register_forward_pre_hook(
            lambda module, inputs: bags.append(tuple(inputs[0].shape)))
        with patch.object(engine, "load_images", side_effect=lambda paths, _: images[paths]), \
                patch("downstream.shared_model._check_shape", side_effect=shape_check_small):
            results.append(helper(model, sample_for(n), micro_batch, device, target, False, diag))
        handle.remove()
        bag_handle.remove()
        assert bags == [(1, n, 512)], bags
        logits.append(captured[0])
        rngs.append(engine.capture_rng_state(device))
        diagnostics.append(diag)
    assert diagnostics[0]["mode"] == "full_graph"
    assert diagnostics[1]["mode"] == "gradpool"
    for key in rngs[0]:
        assert torch.equal(rngs[0][key], rngs[1][key]), key
    torch.testing.assert_close(logits[0], logits[1], atol=0, rtol=0)
    assert results[0] == results[1], results
    prefixes = ("sr.encoder.", "region_encoder.", "wikg.", "mil_head.", "classifier.")
    errors = {prefix: 0.0 for prefix in prefixes}
    reference = dict(full.named_parameters())
    active = {prefix: 0 for prefix in prefixes}
    for name, parameter in cached.named_parameters():
        expected = reference[name].grad
        if expected is None:
            assert parameter.grad is None, name
            continue
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, expected, rtol=3e-4, atol=3e-5, msg=name)
        for prefix in prefixes:
            if name.startswith(prefix):
                errors[prefix] = max(errors[prefix], (parameter.grad - expected).abs().max().item())
                active[prefix] += int(expected.abs().max() > 0)
    # N=1 gives a constant ABMIL weight of one, so its score gradients are zero.
    assert all(count > 0 for prefix, count in active.items() if n > 1 or prefix != "mil_head."), active
    for name, value in cached.named_buffers():
        torch.testing.assert_close(value, dict(full.named_buffers())[name], atol=0, rtol=0, msg=name)
    batch_norms = [module for module in cached.region_encoder.modules() if isinstance(module, nn.BatchNorm2d)]
    assert len(batch_norms) == 20
    assert all(module.training and module.num_batches_tracked.item() == (n + micro_batch - 1) // micro_batch
               for module in batch_norms)
    # One matching Adam update also checks effective gradients near zero.
    for model in (full, cached):
        torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4).step()
    update_error = 0.0
    for name, parameter in cached.named_parameters():
        expected = reference[name]
        update_error = max(update_error, (parameter - expected).abs().max().item())
        torch.testing.assert_close(parameter, expected, rtol=3e-4, atol=3e-6, msg=name)
    return dict(device=str(device), size=size, n=n, micro_batch=micro_batch, dropout=dropout,
                logits_max_abs=(logits[0] - logits[1]).abs().max().item(),
                loss_abs=abs(results[0][0] - results[1][0]), gradient_max_abs=errors,
                adam_parameter_max_abs=update_error, rng_equal=True, buffers_equal=True,
                batch_norm_count=len(batch_norms), bn_updates=(n + micro_batch - 1) // micro_batch)


class WiKGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)

    @staticmethod
    def downstream_only(args=None):
        def initialize(module, sr, model_spec=None):
            nn.Module.__init__(module)
        with patch.object(SharedE2EModel, "__init__", initialize):
            return ResNetWiKGABMIL(None, downstream_args=args or ARGS)

    def test_dynamic_n_forward_backward_through_wikg_abmil_classifier(self):
        model = self.downstream_only()
        for n in range(1, 36):
            with self.subTest(n=n):
                model.zero_grad(set_to_none=True)
                tokens = torch.randn(n, 512, requires_grad=True)
                with patch("torch.topk", wraps=torch.topk) as topk:
                    logits = model.forward_embeddings(tokens)
                self.assertEqual(topk.call_args.kwargs["k"], min(6, n))
                self.assertEqual(tuple(logits.shape), (1, 2))
                F.cross_entropy(logits, torch.tensor([1])).backward()
                self.assertTrue(torch.isfinite(tokens.grad).all())
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_paper_dot_product_matches_independent_neighbor_loop(self):
        module = WiKGCommunication(feature_dim=4, topk=2, dropout=0).double()
        x = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
        projected = module.input_projection(x)
        projected = (projected + projected.mean(1, keepdim=True)) * 0.5
        h, t = module.W_head(projected), module.W_tail(projected)
        bags = []
        for b in range(2):
            nodes = []
            for i in range(3):
                scores = torch.stack([torch.dot(h[b, i], t[b, j]) / 2 for j in range(3)])
                selected = torch.argsort(scores, descending=True)[:2]
                p = torch.softmax(scores[selected], dim=0)
                relations = [p[j] * t[b, index] + (1 - p[j]) * h[b, i]
                             for j, index in enumerate(selected)]
                u = torch.stack([torch.dot(t[b, index], torch.tanh(h[b, i] + relations[j]))
                                 for j, index in enumerate(selected)])
                a = torch.softmax(u, dim=0)
                message = sum(a[j] * t[b, index] for j, index in enumerate(selected))
                nodes.append(module.activation(module.linear1(h[b, i] + message))
                             + module.activation(module.linear2(h[b, i] * message)))
            bags.append(torch.stack(nodes))
        expected = torch.stack(bags)
        actual = module(x)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        params = (x, *module.parameters())
        expected_grad = torch.autograd.grad(expected.square().sum(), params, retain_graph=True)
        actual_grad = torch.autograd.grad(actual.square().sum(), params)
        for left, right in zip(expected_grad, actual_grad):
            torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)

    def test_gradpool_matches_full_graph_real_hat_resnet_and_dropout(self):
        for n, micro_batch in ((1, 1), (3, 1), (3, 2)):
            for dropout in (0.0, 0.3):
                with self.subTest(n=n, micro_batch=micro_batch, dropout=dropout):
                    result = equivalence_probe(dropout, n=n, micro_batch=micro_batch)
                    print("WIKG_EQUIVALENCE " + json.dumps(result), flush=True)

    def test_evaluate_uses_one_whole_bag_communication_and_no_hr(self):
        model = SmallSRWiKG(SmallSR(), downstream_args=ARGS).eval()
        inputs = torch.randn(3, 3, 64, 64)
        bags, sizes = [], []
        handle = model.wikg.register_forward_pre_hook(lambda module, args: bags.append(tuple(args[0].shape)))
        def load(paths, size):
            sizes.append(size)
            return inputs[paths]
        negative = dict(sample_for(3), slide_id="negative", label=0)
        with patch.object(engine, "load_images", side_effect=load), \
                patch("downstream.shared_model._check_shape", side_effect=shape_check_small), \
                patch.object(model, "forward_sr", side_effect=AssertionError("Evaluation must not use SR")):
            result = engine.evaluate(model, [negative, sample_for(3)], "cpu", cls_micro_batch=2)
        handle.remove()
        self.assertEqual(bags, [(1, 3, 512), (1, 3, 512)])
        self.assertEqual(sizes, [256] * 4)
        self.assertEqual(len(result["predictions"]), 2)

    def test_train_wsi_dispatch_one_step_and_combined_sr(self):
        inputs = torch.randn(3, 3, 64, 64)
        initial = SmallSRWiKG(SmallSR(), downstream_args=ARGS)
        updated, results = [], []
        for enabled in (False, True):
            model = copy.deepcopy(initial)
            model.classification_gradpool = enabled
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            helper_name = ("classification_loss_and_backward" if enabled
                           else "full_graph_classification_loss_and_backward")
            torch.manual_seed(509)
            with patch.object(engine, "load_images", side_effect=lambda paths, _: inputs[paths]), \
                    patch("downstream.shared_model._check_shape", side_effect=shape_check_small), \
                    patch.object(engine, helper_name, wraps=getattr(engine, helper_name)) as classify, \
                    patch.object(optimizer, "step", wraps=optimizer.step) as step:
                results.append(engine.train_wsi(model, sample_for(3), optimizer, "cpu",
                                                cls_micro_batch=2, lambda_sr=0.4))
            self.assertEqual(classify.call_count, 1)
            self.assertEqual(step.call_count, 1)
            updated.append(model.state_dict())
        for key in ("loss_cls", "loss_sr", "loss_total", "p_tumor"):
            self.assertEqual(results[0][key], results[1][key])
        for name, value in updated[0].items():
            torch.testing.assert_close(value, updated[1][name], atol=3e-6, rtol=3e-4, msg=name)

    def test_config_registry_signature_and_checkpoint_roundtrip(self):
        root = Path(__file__).resolve().parents[1]
        config = root / "configs/resnet_wikg_abmil.yaml"
        spec = train.load_downstream_spec(config)
        self.assertEqual(spec, dict(name="resnet_wikg_abmil", args=ARGS))
        self.assertIs(get_model_class(spec["name"]), ResNetWiKGABMIL)
        args = train.parse_args(["--model", spec["name"], "--sr-root", ".",
                                 "--sr-config", str(root / "configs/sr_hat_8rhag_scratch.yaml"),
                                 "--downstream-config", str(config)])
        self.assertEqual(args.downstream_spec, spec)
        model = SmallSRWiKG(SmallSR(), downstream_args=ARGS)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "splits_0.csv").write_text("train,val\na,b\n")
            provenance = {key: {"sha256": "test"} for key in ("patch_manifest", "labels")}
            sig = train.experiment_signature(args, provenance, directory, [0], "sr-test")
            self.assertEqual(sig["classification_memory"], "gradient_cache_v1")
            self.assertIn("downstream/resnet_wikg_abmil/model.py", sig["code"])
            changed = copy.deepcopy(args)
            changed.downstream_spec["args"]["classification_gradpool"] = False
            self.assertEqual(train.experiment_signature(changed, provenance, directory, [0], "sr-test")
                             ["classification_memory"], "full_graph_v1")
            path = directory / "best.pth"
            with patch.object(torch.cuda, "get_rng_state", return_value=torch.get_rng_state()):
                train.save_checkpoint(path, model, optimizer, 0, 1, {}, args, provenance, sig)
            saved = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint_model_name(saved), spec["name"])
        self.assertEqual(saved["downstream_spec"], spec)
        restored = SmallSRWiKG(SmallSR(), downstream_args=saved["downstream_spec"]["args"])
        restored.load_state_dict(saved["model_state"], strict=True)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
        with patch("downstream.registry.build_model", return_value=restored) as builder:
            build_from_spec(spec["name"], ".", {}, "cpu", saved["downstream_spec"])
        self.assertEqual(builder.call_args.kwargs["downstream_args"], ARGS)

    def test_config_validation_and_debug_once(self):
        for key, value in (("wikg_topk", 0), ("wikg_topk", True), ("wikg_dropout", float("nan")),
                           ("wikg_dropout", 1), ("classification_gradpool", "false"),
                           ("debug_shapes", "true"), ("unknown", 1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                ResNetWiKGABMIL.validate_downstream_args(dict(ARGS, **{key: value}))
        model = self.downstream_only(dict(ARGS, debug_shapes=True)).eval()
        output = io.StringIO()
        with redirect_stdout(output):
            model.forward_embeddings(torch.randn(6, 512))
            model.forward_embeddings(torch.randn(6, 512))
        self.assertEqual(output.getvalue().count("WiKG input:"), 1)
        for token in ("[1, 6, 512]", "[1, 6, 6]", "[1, 6, 6, 512]", "[1, 512]", "[1, 2]"):
            self.assertIn(token, output.getvalue())
        self.assertIsInstance(model.mil_head, GatedAttentionMIL)
        self.assertEqual((model.classifier.in_features, model.classifier.out_features), (512, 2))


if __name__ == "__main__":
    unittest.main()
