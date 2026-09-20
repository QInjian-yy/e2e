"""CPU checks of real offline evaluation entrypoints and augmentation metadata."""

import copy
import csv
import io
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_engine import TinyE2E
from test_wsi_data import write_csv
import eval_train_checkpoint
from training import evaluate as offline_validation
from training import train
from wsi_data import load_fold_datasets


class AugmentationEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.splits = self.root / "downstream_train"
        self.labels = [dict(case_id=f"case_{i}", slide_id=f"slide_{i}", label=i % 2)
                       for i in range(4)]
        write_csv(self.splits / "camelyon16_labels.csv",
                  ("case_id", "slide_id", "label"), self.labels)
        write_csv(self.splits / "splits_0.csv", ("train", "val"),
                  [dict(train="case_0", val="case_2"),
                   dict(train="case_1", val="case_3")])
        self.raw = {}
        regions = []
        y, x = np.indices((256, 256))
        for index, count in enumerate((3, 1, 2, 3)):
            for region in range(count):
                filename = f"slide_{index}_r{region}.png"
                regions.append(dict(filename=filename, slide_id=f"slide_{index}"))
                array = np.stack((x, y, (3*x + 5*y + 19*index + region) % 256),
                                 axis=-1).astype(np.uint8)
                lr_path = self.root / "images_256" / filename
                hr_path = self.root / "images_8192" / filename
                lr_path.parent.mkdir(exist_ok=True)
                hr_path.parent.mkdir(exist_ok=True)
                Image.fromarray(array).save(lr_path)
                # Evaluating either split must never decode an HR target.
                hr_path.write_bytes(b"HR must not be opened during evaluation")
                self.raw[filename] = torch.from_numpy(array).permute(2, 0, 1).float() / 255
        write_csv(self.root / "manifests" / "patch_manifest.csv",
                  ("filename", "slide_id"), regions)
        self.train_data, self.val_data = load_fold_datasets(self.root, 0)
        self.provenance = train.data_provenance(self.root, self.train_data)
        self.args = SimpleNamespace(
            model="mean_resnet", data_root=self.root, sr_root=self.root / "sr",
            labels_csv=None, split_dir=None, gpu=0, num_workers=0,
            cls_micro_batch=2, sr_micro_batch=1, lr=1e-5, weight_decay=1e-4,
            early_stopping_patience=5, lambda_sr=0.37, seed=7,
            num_folds=1, best_metric="auc",
        )
        self.enabled = dict(enabled=True, horizontal_flip=0.5,
                            vertical_flip=0.5, rotation90=True)

    def signature(self, args):
        return train.experiment_signature(args, self.provenance, self.splits,
                                          [0], "synthetic-sr-config")

    def save_toy_checkpoint(self, model_type, augmentation):
        torch.manual_seed(73)
        model = model_type()
        model.model_spec = {"name": "cpu-evaluation-fixture"}
        with torch.no_grad():
            model.region_encoder[1].running_mean.copy_(torch.tensor([0.2, -0.4, 0.7]))
            model.region_encoder[1].running_var.copy_(torch.tensor([0.8, 1.3, 1.7]))
            model.region_encoder[1].num_batches_tracked.fill_(11)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        # Populate real Adam state without training on a WSI or invoking SR.
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.01)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        args = copy.deepcopy(self.args)
        args.model = model.model_name
        if augmentation is not None:
            args.augmentation = copy.deepcopy(augmentation)
        run_dir = self.root / (model.model_name + ("_old" if augmentation is None else "_aug"))
        checkpoint = run_dir / "fold_0" / "best.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        metrics = dict(best_epoch=1, best_score=0.7, best_auc=0.7)
        with patch.object(torch.cuda, "get_rng_state", return_value=torch.tensor([1], dtype=torch.uint8)):
            train.save_checkpoint(checkpoint, model, optimizer, 0, 1, metrics,
                                  args, self.provenance, self.signature(args))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.assertEqual(saved["format"], train.CHECKPOINT_FORMAT)
        self.assertEqual(set(saved["optimizer_state"]), {"state", "param_groups"})
        self.assertEqual(saved["optimizer_state"]["param_groups"], optimizer.state_dict()["param_groups"])
        self.assertTrue(saved["optimizer_state"]["state"])
        for key, state in optimizer.state_dict()["state"].items():
            self.assertEqual(set(saved["optimizer_state"]["state"][key]), set(state))
            for name, value in state.items():
                torch.testing.assert_close(saved["optimizer_state"]["state"][key][name], value,
                                           atol=0, rtol=0)
        self.assertEqual(set(saved["model_state"]), set(model.state_dict()))
        self.assertEqual("augmentation" in saved["config"], augmentation is not None)
        return run_dir, checkpoint, saved

    def test_offline_train_and_validation_ignore_saved_augmentation_and_preserve_bn(self):
        for model_type in (TinyE2E,):
            for augmentation in (None, self.enabled):
                with self.subTest(model=model_type.__name__, augmentation=augmentation):
                    run_dir, checkpoint, saved = self.save_toy_checkpoint(model_type, augmentation)
                    checkpoint_bytes = checkpoint.read_bytes()
                    for split, entrypoint, dataset in (
                            ("train", eval_train_checkpoint, self.train_data),
                            ("val", offline_validation, self.val_data)):
                        with self.subTest(split=split):
                            models, inputs, loaded = [], [], []

                            def build(model_name, sr_root, model_spec, device):
                                self.assertEqual(model_name, saved["model_name"])
                                self.assertEqual(model_spec, saved["model_spec"])
                                self.assertEqual(device, torch.device("cpu"))
                                model = model_type().train()
                                models.append(model)

                                def after_load(module, incompatible):
                                    self.assertEqual(incompatible.missing_keys, [])
                                    self.assertEqual(incompatible.unexpected_keys, [])
                                    loaded.append({key: value.clone() for key, value in module.state_dict().items()})

                                def before_encode(module, arguments):
                                    self.assertFalse(torch.is_grad_enabled())
                                    self.assertTrue(all(not child.training for child in model.modules()))
                                    self.assertLessEqual(len(arguments[0]), self.args.cls_micro_batch)
                                    inputs.append(arguments[0].detach().clone())

                                model.register_load_state_dict_post_hook(after_load)
                                model.sr.encoder.register_forward_pre_hook(before_encode)
                                return model

                            random_state = random.getstate()
                            output_path = (checkpoint.parent / "train_eval_predictions.csv" if split == "train"
                                           else checkpoint.parent / "val_predictions.csv")
                            with patch.object(entrypoint, "cuda_device", return_value=torch.device("cpu")), \
                                    patch.object(entrypoint, "build_from_spec", side_effect=build), \
                                    redirect_stdout(io.StringIO()):
                                if split == "train":
                                    entrypoint.evaluate_fold(run_dir, 0)
                                else:
                                    entrypoint.main([
                                        "--data-root", str(self.root), "--sr-root", str(self.args.sr_root),
                                        "--checkpoint", str(checkpoint), "--predictions-csv", str(output_path),
                                    ])
                            self.assertEqual(random.getstate(), random_state)
                            self.assertEqual(len(models), 1)
                            self.assertEqual(len(loaded), 1)
                            model = models[0]
                            expected = torch.stack([self.raw[path.name] for sample in dataset.samples
                                                    for path in sample["lr_paths"]])
                            torch.testing.assert_close(torch.cat(inputs), expected, atol=0, rtol=0)
                            self.assertEqual(model.embedding_forward_calls, len(dataset))
                            self.assertEqual(model.sr_calls, [])
                            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
                            for name, value in saved["model_state"].items():
                                torch.testing.assert_close(loaded[0][name], value, atol=0, rtol=0)
                                torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
                            self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
                            with output_path.open(newline="", encoding="utf-8") as handle:
                                predictions = list(csv.DictReader(handle))
                            self.assertEqual([row["slide_id"] for row in predictions],
                                             [sample["slide_id"] for sample in dataset.samples])
                            self.assertEqual([int(row["label"]) for row in predictions],
                                             [sample["label"] for sample in dataset.samples])

    def test_signature_tracks_enabled_augmentation_and_default_matches_disabled(self):
        missing = self.signature(self.args)
        disabled = copy.deepcopy(self.args)
        disabled.augmentation = dict(self.enabled, enabled=False)
        self.assertEqual(self.signature(disabled), missing)
        args = copy.deepcopy(self.args)
        args.augmentation = copy.deepcopy(self.enabled)
        enabled = self.signature(args)
        self.assertEqual(enabled["augmentation"], self.enabled)
        self.assertNotEqual(enabled, missing)
        saved = dict(format=train.CHECKPOINT_FORMAT, model_name="mean_resnet", fold=0, epoch=1,
                     experiment=enabled, data_provenance=self.provenance,
                     metrics=dict(best_epoch=1, best_score=0.7))
        best = copy.deepcopy(saved)
        train.validate_resume_state(saved, best, [{"epoch": "1"}], 0, 2, self.provenance, enabled)
        for key, value in (("horizontal_flip", 0.1), ("vertical_flip", 0.9),
                           ("rotation90", False), ("enabled", False)):
            with self.subTest(setting=key):
                changed = copy.deepcopy(args)
                changed.augmentation[key] = value
                different = self.signature(changed)
                self.assertNotEqual(enabled, different)
                with self.assertRaisesRegex(ValueError, "Resume code/data/hyperparameters"):
                    train.validate_resume_state(saved, best, [{"epoch": "1"}], 0, 2,
                                                self.provenance, different)
        # Existing source-hash restrictions still apply to checkpoints from older code.
        older_code = copy.deepcopy(enabled)
        older_code["code"]["wsi_data.py"] = "previous-source-hash"
        with self.assertRaisesRegex(ValueError, "Resume code/data/hyperparameters"):
            train.validate_resume_state(saved, best, [{"epoch": "1"}], 0, 2,
                                        self.provenance, older_code)


if __name__ == "__main__":
    unittest.main()
