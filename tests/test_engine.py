"""CPU toy-model checks for gradient arithmetic; these are not an 8K CUDA smoke test."""

import copy
import io
import re
import sys
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from downstream.shared_model import _temporary_bn_buffers
from training import engine


class TinyHAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_first = nn.Conv2d(3, 4, 1)

    def forward(self, lr):
        feature = self.conv_first(lr).tanh()
        return F.dropout(feature, p=0.25, training=self.training)


class TinySR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyHAT()
        self.head = nn.Conv2d(4, 3, 1)


class TinyE2E(nn.Module):
    """One shared HAT, with the same callable boundary as the production wrapper."""

    def __init__(self):
        super().__init__()
        self.sr = TinySR()
        self.region_encoder = nn.Sequential(
            nn.Conv2d(4, 3, 1), nn.BatchNorm2d(3), nn.Tanh(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1)
        )
        self.classifier = nn.Linear(3, 2)
        self.model_name = "mean_resnet"
        self.pooling_name = "mean"
        self.sr_calls = []
        self.cls_calls = []
        self.embedding_forward_calls = 0
        self.clear_calls = 0
        self.observe_sr = None

    def _encode(self, lr):
        return self.region_encoder(self.sr.encoder(lr))

    def encode_regions(self, lr, preserve_bn_stats=False, log_shapes=False):
        self.cls_calls.append(len(lr))
        context = (_temporary_bn_buffers(self.region_encoder)
                   if preserve_bn_stats and self.training else nullcontext())
        with context:
            return self._encode(lr)

    def aggregate_embeddings(self, embeddings):
        return embeddings.mean(dim=0, keepdim=True)

    def classify(self, wsi_embedding):
        return self.classifier(wsi_embedding)

    def forward_embeddings(self, embeddings):
        self.embedding_forward_calls += 1
        return self.classify(self.aggregate_embeddings(embeddings))

    def downstream_gradient_groups(self):
        return {"WSI classifier": self.classifier.parameters()}

    def forward_sr(self, lr, log_shapes=False):
        self.sr_calls.append(len(lr))
        if self.observe_sr is not None:
            self.observe_sr()
        return self.sr.head(self.sr.encoder(lr))

    def clear_sr_cache(self):
        self.clear_calls += 1


class CountSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.04)
        self.zero_calls = 0
        self.step_calls = 0

    def zero_grad(self, *args, **kwargs):
        self.zero_calls += 1
        return super().zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


class EngineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(61)
        self.device = torch.device("cpu")
        self.lr = torch.randn(5, 3, 2, 2)
        self.hr = torch.randn(5, 3, 2, 2)
        self.sample = {
            "slide_id": "toy_wsi", "case_id": "toy_case", "label": 1,
            "lr_paths": [Path(f"lr_{i}") for i in range(5)],
            "hr_paths": [Path(f"hr_{i}") for i in range(5)], "n_regions": 5,
        }
        self.loads = []

    def load_images(self, paths, size):
        names = [Path(path).name for path in paths]
        self.loads.append((names, size))
        expected_prefix = "lr_" if size == 256 else "hr_"
        self.assertIn(size, (256, 8192))
        self.assertTrue(all(name.startswith(expected_prefix) for name in names))
        tensor = self.lr if size == 256 else self.hr
        return tensor[[int(name.rsplit("_", 1)[1]) for name in names]].clone()

    def train(self, model, optimizer, verbose=False):
        return engine.train_wsi(
            model, self.sample, optimizer, self.device,
            cls_micro_batch=2, sr_micro_batch=2, lambda_sr=0.37, verbose=verbose,
        )

    def test_phased_gradients_and_single_step_match_combined_loss_with_tail(self):
        for model_type in (TinyE2E,):
            with self.subTest(pooling=model_type.__name__):
                self.loads.clear()
                actual = model_type()
                reference = copy.deepcopy(actual)
                expected_optimizer = CountSGD(reference.parameters())
                expected_optimizer.zero_grad(set_to_none=True)

                torch.manual_seed(907)
                embeddings = torch.cat([reference._encode(chunk) for chunk in self.lr.split(2)])
                cls_loss = F.cross_entropy(
                    reference.forward_embeddings(embeddings), torch.tensor([1])
                )
                sr_output = torch.cat([reference.forward_sr(chunk) for chunk in self.lr.split(2)])
                sr_loss = F.l1_loss(sr_output, self.hr)
                total_loss = cls_loss + 0.37 * sr_loss
                total_loss.backward()
                expected_rng = torch.get_rng_state()
                expected_optimizer.step()

                optimizer = CountSGD(actual.parameters())

                def check_no_early_step_or_zero():
                    self.assertEqual(optimizer.zero_calls, 1)
                    self.assertEqual(optimizer.step_calls, 0)
                    self.assertIsNotNone(actual.sr.encoder.conv_first.weight.grad)
                    self.assertIsNotNone(actual.region_encoder[0].weight.grad)
                    self.assertIsNotNone(actual.classifier.weight.grad)

                actual.observe_sr = check_no_early_step_or_zero
                log = io.StringIO()
                torch.manual_seed(907)
                with patch.object(engine, "load_images", self.load_images), redirect_stdout(log):
                    result = self.train(actual, optimizer, verbose=True)

                self.assertEqual(optimizer.zero_calls, 2)
                self.assertEqual(optimizer.step_calls, 1)
                self.assertEqual(actual.cls_calls, [2, 2, 1, 2, 2, 1])
                self.assertEqual(actual.sr_calls, [2, 2, 1])
                self.assertIn("gradient caching", log.getvalue())
                self.assertIn("classification -> shared HAT incoming gradient", log.getvalue())
                self.assertIn("sr -> shared HAT incoming gradient", log.getvalue())
                self.assertTrue(all(len(paths) <= 2 for paths, _ in self.loads))
                self.assertEqual(sum(len(paths) for paths, size in self.loads if size == 8192), 5)
                self.assertAlmostEqual(result["loss_cls"], cls_loss.item(), places=6)
                self.assertAlmostEqual(result["loss_sr"], sr_loss.item(), places=6)
                self.assertAlmostEqual(result["loss_total"], total_loss.item(), places=6)
                self.assertEqual(result["label"], self.sample["label"])
                self.assertIs(type(result["p_tumor"]), float)
                self.assertIs(type(result["prediction"]), int)
                self.assertFalse(any(isinstance(value, torch.Tensor) for value in result.values()))
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                expected_state = reference.state_dict()
                for name, value in actual.state_dict().items():
                    torch.testing.assert_close(value, expected_state[name], atol=1e-7, rtol=1e-6)

    def test_gradpool_classification_matches_full_graph_reference(self):
        for model_type in (TinyE2E,):
            with self.subTest(model=model_type.__name__):
                torch.manual_seed(101)
                gradpool = model_type().train()
                reference = copy.deepcopy(gradpool).train()
                target = torch.tensor([self.sample["label"]])

                torch.manual_seed(907)
                reference_embeddings = torch.cat(
                    [reference._encode(chunk) for chunk in self.lr.split(2)]
                )
                reference_logits = reference.forward_embeddings(reference_embeddings)
                reference_logits_value = reference_logits.detach().clone()
                reference_loss = F.cross_entropy(reference_logits, target)
                reference_loss.backward()
                reference_rng = torch.get_rng_state().clone()
                replay_embeddings, gradpool_logits = [], []
                hooks = [
                    gradpool.region_encoder.register_forward_hook(
                        lambda module, inputs, output: replay_embeddings.append(
                            output.detach().clone()
                        )
                    ),
                    gradpool.classifier.register_forward_hook(
                        lambda module, inputs, output: gradpool_logits.append(
                            output.detach().clone()
                        )
                    ),
                ]
                torch.manual_seed(907)
                try:
                    with patch.object(engine, "load_images", self.load_images):
                        cls_value, _, _, _ = engine.classification_loss_and_backward(
                            gradpool, self.sample, 2, self.device, target, verbose=False,
                        )
                finally:
                    for hook in hooks:
                        hook.remove()

                chunks = len(list(self.lr.split(2)))
                stage1 = torch.cat(replay_embeddings[:chunks])
                stage2 = torch.cat(replay_embeddings[chunks:])
                torch.testing.assert_close(stage1, reference_embeddings, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(stage2, reference_embeddings, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(stage2, stage1, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(gradpool_logits[0], reference_logits_value,
                                           rtol=1e-6, atol=1e-7)
                self.assertAlmostEqual(cls_value, reference_loss.item(), places=7)
                self.assertTrue(torch.equal(torch.get_rng_state(), reference_rng))

                reference_buffers = dict(reference.region_encoder.named_buffers())
                for name, value in gradpool.region_encoder.named_buffers():
                    torch.testing.assert_close(value, reference_buffers[name], atol=0, rtol=0)

                reference_parameters = dict(reference.named_parameters())
                group_errors = {"HAT": 0.0, "ResNet": 0.0, "classifier": 0.0}
                for name, parameter in gradpool.named_parameters():
                    expected_gradient = reference_parameters[name].grad
                    if expected_gradient is None:
                        self.assertIsNone(parameter.grad, name)
                        continue
                    self.assertIsNotNone(parameter.grad, name)
                    torch.testing.assert_close(parameter.grad, expected_gradient,
                                               rtol=2e-5, atol=2e-6, msg=name)
                    if name.startswith("sr.encoder."):
                        group = "HAT"
                    elif name.startswith("region_encoder."):
                        group = "ResNet"
                    elif name.startswith("classifier."):
                        group = "classifier"
                    else:
                        self.fail(f"Unexpected parameter group: {name}")
                    error = (parameter.grad - expected_gradient).abs().max().item()
                    group_errors[group] = max(group_errors[group], error)

                embedding_error = (stage2 - stage1).abs().max().item()
                logits_error = (gradpool_logits[0] - reference_logits_value).abs().max().item()
                print("{} reference: embedding_max_abs={:.3e} logits_max_abs={:.3e} "
                      "loss_abs={:.3e} grad_max_abs={}".format(
                          model_type.__name__, embedding_error, logits_error,
                          abs(cls_value - reference_loss.item()),
                          {key: "{:.3e}".format(value) for key, value in group_errors.items()}),
                      flush=True)

    def test_gradient_logging_does_not_change_gradients_rng_or_update(self):
        for model_type in (TinyE2E,):
            with self.subTest(model=model_type.__name__):
                torch.manual_seed(109)
                quiet = model_type()
                logged = copy.deepcopy(quiet)
                quiet_optimizer = CountSGD(quiet.parameters())
                logged_optimizer = CountSGD(logged.parameters())

                with patch.object(engine, "load_images", self.load_images):
                    torch.manual_seed(997)
                    quiet_result = self.train(quiet, quiet_optimizer)
                    quiet_rng = torch.get_rng_state().clone()
                    torch.manual_seed(997)
                    output = io.StringIO()
                    with redirect_stdout(output):
                        logged_result = engine.train_wsi(
                            logged, self.sample, logged_optimizer, self.device,
                            cls_micro_batch=2, sr_micro_batch=2, lambda_sr=0.37,
                            gradient_log_context={
                                "fold": 1, "epoch": 2,
                                "fold_wsi_iteration": 10, "epoch_wsi": "3/5",
                            },
                        )
                    logged_rng = torch.get_rng_state().clone()

                timing = {"seconds", "classification_seconds", "sr_seconds",
                          "regions_per_second"}
                self.assertEqual({key: value for key, value in quiet_result.items()
                                  if key not in timing},
                                 {key: value for key, value in logged_result.items()
                                  if key not in timing})
                self.assertTrue(torch.equal(quiet_rng, logged_rng))
                for name, value in quiet.state_dict().items():
                    torch.testing.assert_close(value, logged.state_dict()[name],
                                               atol=0, rtol=0, msg=name)
                text = output.getvalue()
                self.assertIn("[gradient] fold=1 epoch=2 fold_wsi_iteration=10", text)
                self.assertIn("HAT:l2=", text)
                self.assertIn("Gaussian:l2=", text)
                self.assertIn("ResNet18:l2=", text)
                self.assertIn("WSI classifier:l2=", text)
                self.assertIn("cache_boundary mode=gradpool dL_dH:shape=[5, 3]", text)
                self.assertIn("classification:calls=", text)
                self.assertIn("sr:calls=", text)
                self.assertIn("aggregate_l2=", text)
                self.assertIn("cosine=", text)
                consistency = re.search(r"consistency_max_abs=([0-9.eE+-]+)", text)
                self.assertIsNotNone(consistency)
                self.assertLess(float(consistency.group(1)), 1e-7)

    def test_missing_hr_nan_and_oom_never_step_partial_wsi(self):
        for failure in ("missing_hr", "nan_hr", "nan_sr", "oom_sr"):
            with self.subTest(failure=failure):
                model = TinyE2E()
                optimizer = CountSGD(model.parameters())
                before = {name: value.clone() for name, value in model.state_dict().items()}
                original_forward = model.forward_sr

                def forward(lr, log_shapes=False):
                    # Let one complete micro-batch backward first, then fail.
                    if len(model.sr_calls) == 1:
                        if failure == "oom_sr":
                            raise torch.OutOfMemoryError("intentional CPU toy OOM exception")
                        if failure == "nan_sr":
                            return original_forward(lr, log_shapes=log_shapes) * float("nan")
                    return original_forward(lr, log_shapes=log_shapes)

                def load(paths, size):
                    if failure == "missing_hr" and size == 8192 and len(model.sr_calls) == 1:
                        raise FileNotFoundError("intentional missing toy HR")
                    if failure == "nan_hr" and size == 8192 and len(model.sr_calls) == 1:
                        return self.load_images(paths, size) * float("nan")
                    return self.load_images(paths, size)

                model.forward_sr = forward
                error_type, error_message = {
                    "missing_hr": (FileNotFoundError, "intentional missing toy HR"),
                    "nan_hr": (FloatingPointError, "Non-finite SR micro-batch 2"),
                    "nan_sr": (FloatingPointError, "Non-finite SR micro-batch 2"),
                    "oom_sr": (torch.OutOfMemoryError, "intentional CPU toy OOM"),
                }[failure]
                with patch.object(engine, "load_images", load):
                    with self.assertRaisesRegex(error_type, error_message):
                        self.train(model, optimizer)
                self.assertEqual(optimizer.zero_calls, 2)
                self.assertEqual(optimizer.step_calls, 0)
                self.assertGreaterEqual(len(model.sr_calls), 1)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_finite_losses_but_nonfinite_gradients_do_not_step(self):
        model = TinyE2E()
        optimizer = CountSGD(model.parameters())
        before = {name: value.clone() for name, value in model.state_dict().items()}
        hook = model.sr.head.weight.register_hook(lambda gradient: gradient * float("nan"))
        try:
            with patch.object(engine, "load_images", self.load_images):
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    self.train(model, optimizer)
        finally:
            hook.remove()
        self.assertEqual(model.sr_calls, [2, 2, 1])
        self.assertEqual(optimizer.zero_calls, 2)
        self.assertEqual(optimizer.step_calls, 0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_evaluate_uses_only_lr_tumor_probability_and_argmax_prediction(self):
        model = TinyE2E()
        # Exactly p_tumor=0.5: torch.argmax deterministically predicts Normal.
        nn.init.zeros_(model.classifier.weight)
        nn.init.zeros_(model.classifier.bias)
        samples = []
        for index, (start, stop, label) in enumerate(((0, 2, 0), (2, 4, 1), (4, 5, 1))):
            sample = dict(self.sample)
            sample.update(slide_id=f"eval_{index}", label=label, n_regions=stop-start,
                          lr_paths=self.sample["lr_paths"][start:stop],
                          hr_paths=self.sample["hr_paths"][start:stop])
            samples.append(sample)

        original_encode = model.encode_regions

        def encode(lr, preserve_bn_stats=False, log_shapes=False):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(model.training)
            return original_encode(lr, preserve_bn_stats, log_shapes)

        model.encode_regions = encode
        with patch.object(engine, "load_images", self.load_images):
            result = engine.evaluate(model, samples, self.device, cls_micro_batch=2)
        self.assertEqual(model.sr_calls, [])
        self.assertTrue(all(size == 256 for _, size in self.loads))
        self.assertEqual(sum(len(paths) for paths, _ in self.loads), 5)
        self.assertAlmostEqual(result["auc"], 0.5)
        self.assertAlmostEqual(result["acc"], 1/3)
        self.assertAlmostEqual(result["bacc"], 0.5)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_shared_metrics_use_probability_for_auc_and_class_for_accuracy(self):
        metrics = engine.classification_metrics(
            labels=[0, 0, 1, 1],
            probabilities=[0.1, 0.9, 0.8, 0.7],
            predictions=[0, 1, 1, 1],
        )
        self.assertAlmostEqual(metrics["auc"], 0.5)
        self.assertAlmostEqual(metrics["acc"], 0.75)
        self.assertAlmostEqual(metrics["bacc"], 0.75)

    def test_complete_toy_epoch_reports_shared_train_metrics(self):
        samples = []
        for index, label in enumerate((0, 1, 0, 1)):
            sample = dict(self.sample)
            sample.update(slide_id=f"train_{index}", label=label)
            samples.append(sample)

        for model_type in (TinyE2E,):
            with self.subTest(model=model_type.__name__):
                torch.manual_seed(73)
                model = model_type()
                optimizer = CountSGD(model.parameters())
                labels, probabilities, predictions = [], [], []
                with patch.object(engine, "load_images", self.load_images):
                    for sample in samples:
                        result = engine.train_wsi(
                            model, sample, optimizer, self.device,
                            cls_micro_batch=2, sr_micro_batch=2, lambda_sr=0.37,
                        )
                        labels.append(result["label"])
                        probabilities.append(result["p_tumor"])
                        predictions.append(result["prediction"])
                    validation = engine.evaluate(model, samples, self.device, cls_micro_batch=2)

                train_metrics = engine.classification_metrics(labels, probabilities, predictions)
                counts = engine.binary_class_counts(labels, predictions)
                self.assertEqual(optimizer.step_calls, len(samples))
                self.assertEqual(model.embedding_forward_calls, 2 * len(samples))
                self.assertEqual(set(train_metrics), {"auc", "acc", "bacc"})
                self.assertTrue(all(0.0 <= value <= 1.0 for value in train_metrics.values()))
                self.assertEqual(counts["gt_normal"], 2)
                self.assertEqual(counts["gt_tumor"], 2)
                self.assertEqual(counts["pred_normal"] + counts["pred_tumor"], len(samples))
                self.assertEqual(set(validation) - {"predictions", "loss"},
                                 {"auc", "acc", "bacc"})


if __name__ == "__main__":
    unittest.main()
