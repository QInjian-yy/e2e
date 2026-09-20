"""CPU checks of experiment identity, checkpoint consistency and optimizer resume."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import train as train_e2e


class CheckpointStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.splits = self.root / "splits"
        self.splits.mkdir()
        for fold in range(3):
            (self.splits / f"splits_{fold}.csv").write_text(
                f"train,val\ntrain_{fold},val_{fold}\n", encoding="utf-8"
            )
        self.args = SimpleNamespace(
            model="mean_resnet",
            lr=1e-5, weight_decay=1e-4, early_stopping_patience=5,
            lambda_sr=0.37, cls_micro_batch=1, sr_micro_batch=1, seed=7,
            gpu=0, fold="0", epochs=20, data_root=self.root / "data",
            sr_root=self.root / "ContinuousSR", output_dir=self.root / "runs",
            sr_config=self.root / "sr.yaml", resume=None,
            num_folds=3, best_metric="auc",
        )
        self.provenance = {
            "patch_manifest": {"path": "old/patch_manifest.csv", "sha256": "patch-hash"},
            "labels": {"path": "old/labels.csv", "sha256": "labels-hash"},
            "split": {"path": "old/splits_0.csv", "sha256": "split-hash"},
        }
        self.experiment = self.signature()

    def signature(self, args=None, provenance=None, config_hash="config-hash", fold_ids=None):
        return train_e2e.experiment_signature(
            args or self.args, provenance or self.provenance, self.splits,
            fold_ids or list(range(3)), config_hash,
        )

    def test_signature_ignores_paths_and_gpu_but_tracks_training_identity(self):
        moved = copy.deepcopy(self.args)
        moved.gpu, moved.fold, moved.epochs = 3, "2", 30
        moved.data_root = Path("elsewhere/data")
        moved.output_dir = Path("elsewhere/output")
        moved.sr_root = Path("elsewhere/ContinuousSR")
        moved.sr_config = None  # Resume inherits the saved SR configuration hash.
        moved_provenance = copy.deepcopy(self.provenance)
        for entry in moved_provenance.values():
            entry["path"] = "elsewhere/copied.csv"
        self.assertEqual(self.experiment, self.signature(moved, moved_provenance))

        changed = copy.deepcopy(self.args)
        changed.lambda_sr = 0.75
        self.assertNotEqual(self.experiment, self.signature(changed))
        changed = copy.deepcopy(self.args)
        changed.weight_decay = 1e-5
        self.assertNotEqual(self.experiment, self.signature(changed))
        changed = copy.deepcopy(self.args)
        changed.early_stopping_patience = 8
        self.assertNotEqual(self.experiment, self.signature(changed))
        self.assertNotEqual(self.experiment, self.signature(config_hash="different-config-hash"))
        changed_provenance = copy.deepcopy(self.provenance)
        changed_provenance["labels"]["sha256"] = "changed-label-hash"
        self.assertNotEqual(self.experiment, self.signature(provenance=changed_provenance))
        (self.splits / "splits_2.csv").write_text("train,val\nchanged,case\n", encoding="utf-8")
        self.assertNotEqual(self.experiment, self.signature())

    def test_three_fold_protocol_identity_is_stable_for_all_and_single_fold_runs(self):
        args = copy.deepcopy(self.args)
        args.fold = "all"
        self.assertEqual(train_e2e.resolve_fold_plan(args, self.splits), ([0, 1, 2], [0, 1, 2]))
        args.fold = "1"
        self.assertEqual(train_e2e.resolve_fold_plan(args, self.splits), ([0, 1, 2], [1]))

    def fold_rows(self):
        return [
            {"fold": fold, "epochs_completed": 13 + fold, "max_epochs": 20,
             "best_epoch": 13,
             "auc": 0.5 + fold * 0.1, "acc": 0.6 + fold * 0.05,
             "bacc": 0.55 + fold * 0.05, "val_loss": 1.0 - fold * 0.1,
             "experiment": copy.deepcopy(self.experiment)}
            for fold in range(3)
        ]

    def test_summary_matches_sample_std_and_rejects_mixed_folds(self):
        rows = self.fold_rows()
        summary = train_e2e.combine_fold_metrics(rows, list(range(3)))
        self.assertAlmostEqual(summary["auc"]["mean"], 0.6)
        self.assertAlmostEqual(summary["auc"]["std"], np.std([0.5, 0.6, 0.7], ddof=1))
        for mismatch in ("experiment", "epochs", "duplicate", "missing"):
            with self.subTest(mismatch=mismatch):
                changed = copy.deepcopy(rows)
                if mismatch == "experiment":
                    changed[2]["experiment"]["hyperparameters"]["lambda_sr"] = 0.9
                elif mismatch == "epochs":
                    changed[2]["max_epochs"] = 25
                elif mismatch == "duplicate":
                    changed[2]["fold"] = 1
                else:
                    changed.pop()
                with self.assertRaises(ValueError):
                    train_e2e.combine_fold_metrics(changed, list(range(3)))

    def resume_fixtures(self):
        saved = {"format": train_e2e.CHECKPOINT_FORMAT, "model_name": "mean_resnet", "pooling": "mean",
                 "fold": 0, "epoch": 3,
                 "experiment": self.experiment, "data_provenance": self.provenance,
                 "metrics": {"best_epoch": 2, "best_auc": 0.8, "best_score": 0.8}}
        best = {"format": train_e2e.CHECKPOINT_FORMAT, "model_name": "mean_resnet", "pooling": "mean",
                "fold": 0, "epoch": 2,
                "experiment": self.experiment, "data_provenance": self.provenance,
                "metrics": {"auc": 0.8, "best_auc": 0.8, "best_score": 0.8}}
        return copy.deepcopy(saved), copy.deepcopy(best), [{"epoch": str(i)} for i in range(1, 4)]

    def test_resume_accepts_matching_state_and_rejects_inconsistent_artifacts(self):
        saved, best, history = self.resume_fixtures()
        train_e2e.validate_resume_state(saved, best, history, 0, 5, self.provenance, self.experiment)
        for mismatch in ("best_epoch", "best_auc", "best_fold", "best_data", "history_gap",
                         "history_duplicate", "no_new_epoch", "older_epoch", "experiment",
                         "already_stopped"):
            with self.subTest(mismatch=mismatch):
                saved, best, history = self.resume_fixtures()
                epochs = 5
                if mismatch == "best_epoch":
                    best["epoch"] = 4
                elif mismatch == "best_auc":
                    best["metrics"]["auc"] = 0.9
                    best["metrics"]["best_score"] = 0.9
                elif mismatch == "best_fold":
                    best["fold"] = 1
                elif mismatch == "best_data":
                    best["data_provenance"]["labels"]["sha256"] = "changed"
                elif mismatch == "history_gap":
                    history.pop(1)
                elif mismatch == "history_duplicate":
                    history.insert(1, {"epoch": "1"})
                elif mismatch == "no_new_epoch":
                    epochs = 3
                elif mismatch == "older_epoch":
                    epochs = 2
                elif mismatch == "already_stopped":
                    saved["metrics"]["early_stopped"] = True
                else:
                    saved["experiment"]["sr_initialization"]["config_sha256"] = "different"
                with self.assertRaises(ValueError):
                    train_e2e.validate_resume_state(
                        saved, best, history, 0, epochs, self.provenance, self.experiment
                    )

    def test_checkpoint_roundtrip_restores_adam_state_and_next_update(self):
        torch.manual_seed(31)
        model = nn.Linear(3, 2)
        model.model_spec = {"name": "cpu-toy"}
        model.model_name = "mean_resnet"
        model.pooling_name = "mean"
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                     weight_decay=self.args.weight_decay)
        inputs = torch.randn(4, 3)

        def step(network, optim):
            optim.zero_grad(set_to_none=True)
            network(inputs).square().mean().backward()
            optim.step()

        step(model, optimizer)
        cuda_rng_stub = torch.tensor([1, 2, 3], dtype=torch.uint8)
        torch_rng = torch.get_rng_state().clone()
        path = self.root / "last.pth"
        with patch.object(torch.cuda, "get_rng_state", return_value=cuda_rng_stub) as cuda_state:
            train_e2e.save_checkpoint(
                path, model, optimizer, 0, 3, {"auc": 0.8, "best_auc": 0.8, "best_epoch": 3},
                self.args, self.provenance, self.experiment,
            )
        cuda_state.assert_called_once_with()
        self.assertTrue(path.is_file())
        self.assertFalse(path.with_suffix(".pth.tmp").exists())
        saved = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(saved["format"], train_e2e.CHECKPOINT_FORMAT)
        self.assertEqual(saved["model_name"], "mean_resnet")
        self.assertEqual(saved["pooling"], "mean")
        self.assertEqual(saved["experiment"], self.experiment)
        self.assertEqual(saved["config"]["output_dir"], str(self.args.output_dir))
        self.assertEqual(saved["config"]["sr_config"], str(self.args.sr_config))
        torch.testing.assert_close(saved["rng"]["torch"], torch_rng, atol=0, rtol=0)
        torch.testing.assert_close(saved["rng"]["cuda"], cuda_rng_stub, atol=0, rtol=0)

        resumed = nn.Linear(3, 2)
        resumed.load_state_dict(saved["model_state"], strict=True)
        resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=self.args.lr,
                                             weight_decay=self.args.weight_decay)
        resumed_optimizer.load_state_dict(saved["optimizer_state"])
        self.assertEqual(len(resumed_optimizer.state), 2)
        self.assertEqual(resumed_optimizer.param_groups[0]["weight_decay"], 1e-4)
        step(model, optimizer)
        step(resumed, resumed_optimizer)
        for actual, expected in zip(resumed.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_weight_free_eight_rhag_config_and_cli(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "sr_hat_8rhag_scratch.yaml"
        spec = train_e2e.load_sr_model_spec(config_path)
        hat = spec["args"]["encoder_spec"]["args"]
        self.assertEqual(spec["name"], "continuous-gaussian")
        self.assertEqual(hat["depths"], [6] * 8)
        self.assertEqual(hat["num_heads"], [6] * 8)
        self.assertEqual(hat["embed_dim"], 180)
        self.assertNotIn("sd", spec)

        args = train_e2e.parse_args([
            "--model", "mean_resnet", "--sr-root", str(self.args.sr_root),
            "--sr-config", str(config_path),
        ])
        self.assertEqual(args.model, "mean_resnet")
        self.assertEqual(args.sr_config, config_path)
        self.assertEqual(args.weight_decay, 1e-4)
        self.assertEqual(args.early_stopping_patience, 5)
        self.assertFalse(hasattr(args, "sr_checkpoint"))

    def test_early_stopping_requires_five_consecutive_non_improvements(self):
        best, bad = -float("inf"), 0
        stops = []
        for auc in (0.60, 0.59, 0.58, 0.60, 0.57, 0.56):
            best, bad, stop = train_e2e.early_stopping_step(auc, best, bad, patience=5)
            stops.append(stop)
        self.assertEqual(best, 0.60)
        self.assertEqual(bad, 5)
        self.assertEqual(stops, [False, False, False, False, False, True])

        best, bad, stop = train_e2e.early_stopping_step(0.61, best, bad, patience=5)
        self.assertEqual((best, bad, stop), (0.61, 0, False))


if __name__ == "__main__":
    unittest.main()
