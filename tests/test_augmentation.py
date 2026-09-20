"""Paired geometry and WSI isolation on CPU; no production or CUDA training."""

import copy
import io
import pickle
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from augmentation import (AugmentationConfig, AugmentedPath, TransformParams,
                          apply_to_pair, generate_transform_params,
                          load_augmentation_config)
from training import engine
from training.train import parse_args
from wsi_data import Camelyon16WSI, collate_one_wsi, load_fold_datasets, load_images
from test_engine import CountSGD, TinyE2E
from test_wsi_data import write_csv


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"enabled": True, "horizontal_flip": 0.5,
                       "vertical_flip": 0.5, "rotation90": True}

    def write_image(self, filename, array):
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(path)
        return path

    def toy_samples(self):
        samples = []
        pixels = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        for slide, label in enumerate((0, 1)):
            lr_paths, hr_paths = [], []
            for region in range(5):
                array = pixels + slide * 30 + region * 4
                lr_paths.append(self.write_image(f"lr_{slide}_{region}.png", array))
                hr_paths.append(self.write_image(f"hr_{slide}_{region}.png", array))
            samples.append({"slide_id": f"slide_{slide}", "case_id": f"case_{slide}",
                            "label": label, "lr_paths": lr_paths, "hr_paths": hr_paths,
                            "n_regions": len(lr_paths)})
        return samples

    def test_all_16_geometries_preserve_lr_hr_relation_at_two_scales(self):
        # The first fixture preserves the production 32x LR/HR scale ratio.
        for side, scale in ((3, 32), (5, 4)):
            array = np.arange(side * side * 3, dtype=np.uint8).reshape(side, side, 3)
            lr = self.write_image("pair_lr.png", array)
            hr = self.write_image("pair_hr.png", array.repeat(scale, 0).repeat(scale, 1))
            for horizontal in (False, True):
                for vertical in (False, True):
                    for rotation in range(4):
                        with self.subTest(side=side, scale=scale, horizontal=horizontal,
                                          vertical=vertical, rotation=rotation):
                            params = TransformParams(horizontal, vertical, rotation)
                            lr_ref, hr_ref = apply_to_pair(lr, hr, params)
                            self.assertIs(lr_ref.params, hr_ref.params)
                            self.assertEqual(Path(lr_ref), lr)
                            actual_lr = load_images([lr_ref], side)
                            actual_hr = load_images([hr_ref], side * scale)
                            expected = array
                            if horizontal:
                                expected = np.flip(expected, axis=1)
                            if vertical:
                                expected = np.flip(expected, axis=0)
                            expected = np.rot90(expected, rotation).copy()
                            expected = torch.from_numpy(expected).permute(2, 0, 1).float() / 255
                            torch.testing.assert_close(actual_lr[0], expected, atol=0, rtol=0)
                            expanded = actual_lr.repeat_interleave(scale, 2).repeat_interleave(scale, 3)
                            torch.testing.assert_close(actual_hr, expanded, atol=0, rtol=0)
                            self.assertEqual(actual_lr.device.type, "cpu")
                            self.assertEqual(actual_hr.dtype, torch.float32)
                            self.assertTrue(actual_lr.is_contiguous())

    def test_one_draw_per_pair_is_lazy_and_fixed_across_repeated_decodes(self):
        samples = self.toy_samples()
        original = copy.deepcopy(samples)
        dataset = Camelyon16WSI(samples, augmentation=self.config)
        params = [TransformParams(bool(i % 2), bool(i % 3), i % 4) for i in range(5)]
        with patch("wsi_data.generate_transform_params", side_effect=params) as generate:
            with patch("wsi_data.Image.open", side_effect=AssertionError("eager decode")):
                sample = dataset[0]
            self.assertEqual(generate.call_count, 5)
            with patch("wsi_data.generate_transform_params", side_effect=AssertionError("resampled")):
                first = load_images(sample["lr_paths"], 4)
                replay = load_images(sample["lr_paths"], 4)
                target = load_images(sample["hr_paths"], 4)
        torch.testing.assert_close(first, replay, atol=0, rtol=0)
        torch.testing.assert_close(first, target, atol=0, rtol=0)
        self.assertEqual(samples, original)
        self.assertEqual(set(sample), set(samples[0]))
        for index, (lr, hr) in enumerate(zip(sample["lr_paths"], sample["hr_paths"])):
            self.assertIs(lr.params, params[index])
            self.assertIs(lr.params, hr.params)
        second_params = [TransformParams(False, False, 0)] * 5
        with patch("wsi_data.generate_transform_params", side_effect=second_params):
            second = dataset[0]
        self.assertEqual(second["lr_paths"][0].params, second_params[0])
        self.assertEqual(sample["lr_paths"][0].params, params[0])

    def test_disabled_augmentation_preserves_original_sample_and_rng(self):
        samples = self.toy_samples()
        for config in (None, {"enabled": False}):
            with self.subTest(config=config):
                dataset = Camelyon16WSI(samples, augmentation=config)
                python_rng, numpy_rng = random.getstate(), np.random.get_state()
                torch_rng = torch.get_rng_state().clone()
                with patch("wsi_data.generate_transform_params", side_effect=AssertionError("RNG draw")):
                    self.assertIs(dataset[0], samples[0])
                    self.assertIs(dataset.evaluation_view()[0], samples[0])
                self.assertEqual(generate_transform_params(AugmentationConfig()),
                                 TransformParams(False, False, 0))
                self.assertEqual(python_rng, random.getstate())
                current_numpy = np.random.get_state()
                self.assertEqual(numpy_rng[0], current_numpy[0])
                np.testing.assert_array_equal(numpy_rng[1], current_numpy[1])
                self.assertEqual(numpy_rng[2:], current_numpy[2:])
                self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))

    def test_fold_counts_labels_and_evaluation_isolation(self):
        labels, regions = [], []
        for slide in range(4):
            labels.append({"case_id": f"case_{slide}", "slide_id": f"slide_{slide}",
                           "label": slide % 2})
            for region in range(slide + 1):
                filename = f"slide_{slide}_r{region}.png"
                regions.append({"filename": filename, "slide_id": f"slide_{slide}",
                                "split": "train"})
                for directory in ("images_256", "images_8192"):
                    self.write_image(f"{directory}/{filename}", np.zeros((4, 4, 3), np.uint8))
        downstream = self.root / "downstream_train"
        write_csv(downstream / "camelyon16_labels.csv", ("case_id", "slide_id", "label"), labels)
        write_csv(downstream / "splits_0.csv", ("train", "val"),
                  [{"train": "case_0", "val": "case_2"}, {"train": "case_1", "val": "case_3"}])
        write_csv(self.root / "manifests" / "patch_manifest.csv",
                  ("filename", "slide_id", "split"), regions)
        plain_train, plain_val = load_fold_datasets(self.root, 0)
        train, val = load_fold_datasets(self.root, 0, augmentation=self.config)
        train_eval = train.evaluation_view()
        self.assertTrue(train.augmentation.enabled)
        self.assertFalse(val.augmentation.enabled)
        self.assertFalse(train_eval.augmentation.enabled)
        self.assertIs(train_eval.samples, train.samples)
        self.assertEqual(train_eval.labels_csv, train.labels_csv)
        self.assertEqual(train_eval.split_csv, train.split_csv)
        for original, dataset in ((plain_train, train), (plain_val, val), (plain_train, train_eval)):
            self.assertEqual(len(original), len(dataset))
            self.assertEqual(sum(s["n_regions"] for s in original),
                             sum(s["n_regions"] for s in dataset))
            for before, after in zip(original, dataset):
                self.assertEqual(set(before), set(after))
                for key in ("slide_id", "case_id", "label", "n_regions"):
                    self.assertEqual(before[key], after[key])
                for key in ("lr_paths", "hr_paths"):
                    self.assertEqual(before[key], [Path(path) for path in after[key]])
        with patch("wsi_data.generate_transform_params", side_effect=AssertionError("eval augmentation")):
            self.assertIs(val[0], val.samples[0])
            self.assertIs(train_eval[0], train.samples[0])

    def test_augmented_paths_survive_pickle_and_worker_dataloader(self):
        dataset = Camelyon16WSI(self.toy_samples(), augmentation=self.config)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
                            collate_fn=collate_one_wsi, multiprocessing_context="spawn")
        samples = list(loader)
        self.assertEqual(len(samples), 2)
        for sample, original in zip(samples, dataset.samples):
            self.assertEqual(sample["slide_id"], original["slide_id"])
            self.assertEqual(sample["n_regions"], 5)
            for lr, hr in zip(sample["lr_paths"], sample["hr_paths"]):
                self.assertIsInstance(lr, AugmentedPath)
                self.assertEqual(lr, pickle.loads(pickle.dumps(lr)))
                self.assertIs(lr.params, hr.params)
            torch.testing.assert_close(load_images(sample["lr_paths"], 4),
                                       load_images(sample["hr_paths"], 4), atol=0, rtol=0)

    def test_mean_gradpool_replay_micro_batches_and_one_step(self):
        samples = self.toy_samples()
        for model_type in (TinyE2E,):
            for enabled in (False, True):
                with self.subTest(model=model_type.__name__, enabled=enabled):
                    torch.manual_seed(73)
                    model = model_type()
                    optimizer = CountSGD(model.parameters())
                    dataset = Camelyon16WSI(samples, augmentation=dict(self.config, enabled=enabled))
                    sample = dataset[0]
                    loads = []

                    def decode(paths, size):
                        self.assertIn(size, (256, 8192))
                        tensor = load_images(paths, 4)  # Only the toy size differs from production.
                        self.assertEqual(tensor.device.type, "cpu")
                        loads.append((size, tensor.clone()))
                        return tensor

                    with patch.object(engine, "load_images", decode):
                        with patch("wsi_data.generate_transform_params", side_effect=AssertionError("replay draw")):
                            result = engine.train_wsi(model, sample, optimizer, torch.device("cpu"),
                                                      cls_micro_batch=2, sr_micro_batch=2, lambda_sr=0.37)
                    self.assertEqual(optimizer.step_calls, 1)
                    self.assertEqual(optimizer.zero_calls, 2)
                    self.assertEqual(model.cls_calls, [2, 2, 1, 2, 2, 1])
                    self.assertEqual(model.sr_calls, [2, 2, 1])
                    self.assertEqual(model.region_encoder[1].num_batches_tracked.item(), 3)
                    lr_loads = [tensor for size, tensor in loads if size == 256]
                    hr_loads = [tensor for size, tensor in loads if size == 8192]
                    self.assertEqual([len(tensor) for tensor in lr_loads], [2, 2, 1] * 3)
                    self.assertEqual([len(tensor) for tensor in hr_loads], [2, 2, 1])
                    for index in range(3):
                        for actual in (lr_loads[index + 3], lr_loads[index + 6], hr_loads[index]):
                            torch.testing.assert_close(lr_loads[index], actual, atol=0, rtol=0)
                    self.assertTrue(np.isfinite(result["loss_total"]))
                    self.assertEqual(result["label"], sample["label"])
                    self.assertEqual(result["n_regions"], 5)

    def test_train_evaluation_has_no_augmentation_gradients_or_bn_updates(self):
        samples = self.toy_samples()
        augmented = Camelyon16WSI(samples, augmentation=self.config)
        for model_type in (TinyE2E,):
            with self.subTest(model=model_type.__name__):
                model = model_type().train()
                before = {name: value.clone() for name, value in model.state_dict().items()}
                observed = []

                def observe(module, inputs):
                    self.assertFalse(module.training)
                    self.assertFalse(torch.is_grad_enabled())
                    observed.append(inputs[0].clone())

                hook = model.region_encoder.register_forward_pre_hook(observe)

                def decode(paths, size):
                    self.assertEqual(size, 256)
                    self.assertTrue(all(not isinstance(path, AugmentedPath) for path in paths))
                    return load_images(paths, 4)

                results = []
                try:
                    for dataset in (Camelyon16WSI(samples), augmented.evaluation_view()):
                        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_one_wsi)
                        with patch.object(engine, "load_images", decode):
                            with patch("wsi_data.generate_transform_params", side_effect=AssertionError("eval draw")):
                                results.append(engine.evaluate(model, loader, "cpu", cls_micro_batch=2))
                        for name, value in model.state_dict().items():
                            torch.testing.assert_close(value, before[name], atol=0, rtol=0, msg=name)
                finally:
                    hook.remove()
                self.assertEqual(results[0], results[1])
                self.assertEqual(len(observed), 12)
                self.assertEqual(model.sr_calls, [])
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_config_validation_probability_endpoints_and_seed_reproducibility(self):
        for key, values in {"enabled": ("true", 1), "rotation90": ("false", 0),
                            "horizontal_flip": (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"),
                            "vertical_flip": (-0.1, 1.1, float("nan"), float("inf"), True, "0.5")}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises((ValueError, TypeError)):
                    AugmentationConfig(**{key: value})
        self.assertEqual(generate_transform_params(AugmentationConfig(True, 0, 1, False)),
                         TransformParams(False, True, 0))
        self.assertEqual(generate_transform_params(AugmentationConfig(True, 1, 0, False)),
                         TransformParams(True, False, 0))
        random.seed(97)
        first = [generate_transform_params(AugmentationConfig(**self.config)) for _ in range(100)]
        random.seed(97)
        second = [generate_transform_params(AugmentationConfig(**self.config)) for _ in range(100)]
        self.assertEqual(first, second)
        self.assertEqual({params.rotation_k for params in first}, {0, 1, 2, 3})

    def test_cli_defaults_and_yaml_switch(self):
        sr_config = self.root / "sr.yaml"
        sr_config.write_text("model: {}\n", encoding="utf-8")
        argv = ["--sr-root", str(self.root), "--sr-config", str(sr_config)]
        self.assertEqual(parse_args(argv).augmentation, asdict(AugmentationConfig()))
        path = self.root / "augmentation.yaml"
        path.write_text("augmentation:\n  enabled: true\n  horizontal_flip: 0.25\n"
                        "  vertical_flip: 0.75\n  rotation90: false\n", encoding="utf-8")
        expected = {"enabled": True, "horizontal_flip": 0.25,
                    "vertical_flip": 0.75, "rotation90": False}
        self.assertEqual(asdict(load_augmentation_config(path)), expected)
        self.assertEqual(parse_args(argv + ["--augmentation-config", str(path)]).augmentation, expected)
        path.write_text("augmentation:\n  enabled: false\n", encoding="utf-8")
        self.assertFalse(parse_args(argv + ["--augmentation-config", str(path)]).augmentation["enabled"])
        for invalid in ("enabled: true\n", "augmentation: []\n",
                        "augmentation:\n  enabled: true\n  rotation: 45\n"):
            path.write_text(invalid, encoding="utf-8")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv + ["--augmentation-config", str(path)])


if __name__ == "__main__":
    unittest.main()
