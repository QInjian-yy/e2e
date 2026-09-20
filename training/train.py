"""CAMELYON16 E2E trainer for the Mean-ResNet downstream model."""

import argparse
import copy
import csv
import gc
import hashlib
import inspect
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from augmentation import load_augmentation_config
from downstream.registry import (available_models, build_from_spec, build_scratch_model,
                                 checkpoint_model_name, get_model_class)
from plot_training import export_history_json, plot_run_directory
from training.engine import binary_class_counts, evaluate, train_wsi
from wsi_data import DEFAULT_DATA_ROOT, collate_one_wsi, discover_fold_ids, load_fold_datasets

CHECKPOINT_FORMAT = "c16-e2e-gradpool-v2"

HISTORY_FIELDS = (
    "epoch", "train_loss_cls", "train_loss_sr", "train_loss_total",
    "train_auc", "train_acc", "train_bacc",
    "val_loss", "val_auc", "val_acc", "val_bacc",
)
GRADIENT_LOG_INTERVAL = 10


def cuda_device(gpu):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch and the original gsplat environment are required; no CPU fallback")
    torch.cuda.set_device(gpu)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training entrypoint requires BF16 support (A800 recommended)")
    device = torch.device("cuda", gpu)
    properties = torch.cuda.get_device_properties(device)
    print(f"GPU={properties.name}; total={properties.total_memory / 1024**3:.3f} GiB; "
          f"PyTorch={torch.__version__}; CUDA={torch.version.cuda}", flush=True)
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def data_provenance(data_root, dataset):
    paths = {"patch_manifest": Path(data_root) / "manifests" / "patch_manifest.csv",
             "labels": dataset.labels_csv, "split": dataset.split_csv}
    return {name: {"path": str(path.resolve()),
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in paths.items()}


def provenance_hashes(provenance):
    return {name: entry["sha256"] for name, entry in provenance.items()}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sr_model_spec(path):
    """Read a weight-free ContinuousGaussian/HAT model specification."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("--sr-config must contain a top-level 'model' mapping")
    spec = copy.deepcopy(config["model"])
    if "sd" in spec:
        raise ValueError("--sr-config must not contain model weights; scratch initialization only")
    if spec.get("name") != "continuous-gaussian":
        raise ValueError("--sr-config model.name must be 'continuous-gaussian'")
    try:
        encoder = spec["args"]["encoder_spec"]
        hat = encoder["args"]
        depths = hat["depths"]
        heads = hat["num_heads"]
    except (KeyError, TypeError) as exc:
        raise ValueError("--sr-config must define model.args.encoder_spec.args depths/num_heads") from exc
    if encoder.get("name") != "hat":
        raise ValueError("--sr-config encoder_spec.name must be 'hat'")
    if not isinstance(depths, list) or not depths or not all(isinstance(value, int) and value > 0 for value in depths):
        raise ValueError("HAT depths must be a non-empty list of positive integers")
    if (not isinstance(heads, list) or len(heads) != len(depths)
            or not all(isinstance(value, int) and value > 0 for value in heads)):
        raise ValueError("HAT num_heads must be a positive-integer list matching depths")
    embed_dim = hat.get("embed_dim")
    if not isinstance(embed_dim, int) or embed_dim <= 0 or any(embed_dim % value for value in heads):
        raise ValueError("HAT embed_dim must be positive and divisible by every num_heads value")
    if hat.get("upscale") != 4 or hat.get("upsampler") != "pixelshuffle":
        raise ValueError("E2E Gaussian requires HAT upscale=4 and upsampler=pixelshuffle")
    return spec


def serialized_config(args):
    """Convert path arguments into checkpoint-safe strings."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def experiment_signature(args, provenance, split_directory, fold_ids, sr_config_sha256):
    """Common fold identity; include shared infrastructure and the selected model file."""
    code_root = Path(__file__).resolve().parents[1]
    model_source = Path(inspect.getfile(get_model_class(args.model))).resolve()
    code_sources = [
        Path(__file__).resolve(),
        code_root / "training" / "engine.py",
        code_root / "downstream" / "shared_model.py",
        code_root / "downstream" / "registry.py",
        model_source,
        code_root / "wsi_data.py",
        code_root / "augmentation.py",
        code_root / "plot_training.py",
    ]
    sr_sources = sorted((args.sr_root / "models").glob("*.py"))
    if (args.sr_root / "utils.py").is_file():
        sr_sources.append(args.sr_root / "utils.py")
    signature = {
        "sr_initialization": {"kind": "scratch", "config_sha256": sr_config_sha256},
        "downstream": args.model,
        "classification_memory": "gradient_cache_v1",
        "data": {key: provenance[key]["sha256"] for key in ("patch_manifest", "labels")},
        "splits": [file_sha256(split_directory / "splits_{}.csv".format(fold)) for fold in fold_ids],
        "hyperparameters": {key: getattr(args, key) for key in
                            ("lr", "weight_decay", "early_stopping_patience", "lambda_sr",
                             "cls_micro_batch", "sr_micro_batch", "seed", "num_folds",
                             "best_metric")},
        "code": {path.relative_to(code_root).as_posix(): file_sha256(path)
                 for path in code_sources},
        "sr_code": {path.relative_to(args.sr_root).as_posix(): file_sha256(path)
                    for path in sr_sources},
    }
    augmentation = getattr(args, "augmentation", {})
    if augmentation.get("enabled", False):
        signature["augmentation"] = augmentation
    return signature


def validate_resume_state(saved, best, history_rows, fold, epochs, provenance, experiment):
    expected_model = experiment["downstream"]
    if saved.get("format") != CHECKPOINT_FORMAT or checkpoint_model_name(saved) != expected_model:
        raise ValueError("last.pth has the wrong checkpoint format or downstream model")
    if epochs <= saved["epoch"]:
        raise ValueError(f"--epochs must exceed completed epoch {saved['epoch']} when resuming")
    if saved["metrics"].get("early_stopped", False):
        raise ValueError("This fold already reached its validation-AUC early-stopping condition")
    if saved["experiment"] != experiment:
        raise ValueError("Resume code/data/hyperparameters/initial SR identity differ from last.pth")
    if (best.get("format") != CHECKPOINT_FORMAT
            or checkpoint_model_name(best) != expected_model
            or best["fold"] != fold
            or best["experiment"] != experiment
            or provenance_hashes(best["data_provenance"]) != provenance_hashes(provenance)
            or best["epoch"] != saved["metrics"]["best_epoch"]
            or best["metrics"].get("best_score", best["metrics"].get("best_auc"))
            != saved["metrics"].get("best_score", saved["metrics"].get("best_auc"))):
        raise ValueError("best.pth and last.pth are inconsistent; keep a matching checkpoint pair")
    if [int(row["epoch"]) for row in history_rows] != list(range(1, saved["epoch"] + 1)):
        raise ValueError("history.csv is missing or inconsistent with last.pth; do not append a false history")


def combine_fold_metrics(rows, fold_ids):
    expected = set(fold_ids)
    if {row["fold"] for row in rows} != expected or len(rows) != len(fold_ids):
        raise ValueError("Summary requires exactly folds {}; got {}".format(
            sorted(fold_ids), sorted(row["fold"] for row in rows)))
    if any(row["experiment"] != rows[0]["experiment"] for row in rows[1:]):
        raise ValueError("Refusing to mix different data, initial SR, code or hyperparameters across folds")
    budgets = {row.get("max_epochs", row["epochs_completed"]) for row in rows}
    if len(budgets) != 1:
        raise ValueError("Cross-validation summary requires the same maximum epoch budget for all folds")
    return {key: {"mean": float(np.mean([row[key] for row in rows])),
                  "std": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0}
            for key in ("auc", "acc", "bacc", "val_loss")}


def save_checkpoint(path, model, optimizer, fold, epoch, metrics, args, provenance, experiment):
    saved = {
        "format": CHECKPOINT_FORMAT,
        "model_name": model.model_name,
        "pooling": model.pooling_name,
        "model_spec": model.model_spec,
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "fold": fold, "epoch": epoch, "metrics": metrics,
        "config": serialized_config(args),
        "data_provenance": provenance,
        "experiment": experiment,
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda},
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(saved, temporary)
    temporary.replace(path)


def resolve_fold_plan(args, split_directory):
    available = discover_fold_ids(split_directory)
    num_folds = len(available) if args.num_folds is None else args.num_folds
    if num_folds < 1 or num_folds > len(available):
        raise ValueError("--num-folds must be between 1 and {}; found {}".format(
            len(available), available))
    cv_folds = available[:num_folds]
    if args.fold == "all":
        return cv_folds, cv_folds
    fold = int(args.fold)
    if fold not in cv_folds:
        raise ValueError("--fold must be one of {} or 'all'; got {}".format(cv_folds, fold))
    return cv_folds, [fold]


def validation_metric(validation, metric):
    return validation["loss"] if metric == "val_loss" else validation[metric]


def is_better(metric, current, best):
    return current > best if metric == "auc" else current < best


def early_stopping_step(current_auc, best_auc, bad_epochs, patience):
    """Update strict val-AUC patience; a tied AUC is not an improvement."""
    if current_auc > best_auc:
        return current_auc, 0, False
    bad_epochs += 1
    return best_auc, bad_epochs, bad_epochs >= patience


def parse_args(argv=None, default_model=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=available_models(),
                        default=default_model or "mean_resnet",
                        help="Independent downstream model implementation")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--sr-root", type=Path, required=True)
    parser.add_argument("--sr-config", type=Path,
                        help="Weight-free HAT+Gaussian YAML configuration; required unless --resume")
    parser.add_argument("--augmentation-config", type=Path,
                        help="Optional paired CPU augmentation YAML; disabled by default")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fold", default="all", help="Fold id or 'all' (default: all three folds)")
    parser.add_argument("--num-folds", type=int, default=3,
                        help="Cross-validation protocol size (default: 3; sorted splits_*.csv)")
    parser.add_argument("--best-metric", choices=("auc", "val_loss"), default="auc",
                        help="Validation metric for best.pth selection (higher AUC or lower val_loss)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Adam L2 weight decay shared by all trainable E2E parameters")
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                        help="Stop a fold after this many consecutive epochs without val-AUC improvement")
    parser.add_argument("--lambda-sr", type=float, default=1.0)
    parser.add_argument("--cls-micro-batch", type=int, default=1)
    parser.add_argument("--sr-micro-batch", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Workers return paths only; HR decoding stays in the current SR micro-batch")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Exactly one complete WSI iteration; includes all regions and optimizer.step")
    parser.add_argument("--resume", type=Path,
                        help="Resume the same fold at the next epoch from its complete last.pth")
    args = parser.parse_args(argv)
    try:
        args.augmentation = asdict(load_augmentation_config(args.augmentation_config))
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parents[1] / "runs" / args.model
    if (args.epochs < 1 or args.early_stopping_patience < 1
            or args.cls_micro_batch < 1 or args.num_workers < 0):
        parser.error("epochs/patience/cls-micro-batch must be positive, num-workers must be nonnegative")
    if (not np.isfinite(args.lr) or args.lr <= 0
            or not np.isfinite(args.weight_decay) or args.weight_decay < 0
            or not np.isfinite(args.lambda_sr) or args.lambda_sr <= 0):
        parser.error("lr/lambda-sr must be finite and positive; weight-decay must be finite and nonnegative")
    if args.resume is None and (args.sr_config is None or not args.sr_config.is_file()):
        parser.error("Provide an existing --sr-config to initialize HAT + Gaussian from scratch")
    if args.resume is not None and args.sr_config is not None and not args.sr_config.is_file():
        parser.error("--sr-config does not exist")
    if args.resume is not None and (args.fold == "all" or args.smoke_test):
        parser.error("--resume requires one explicit fold and cannot be combined with --smoke-test")
    if args.resume is not None and not args.resume.is_file():
        parser.error("--resume checkpoint does not exist")
    if args.num_folds is not None and args.num_folds < 1:
        parser.error("--num-folds must be positive")
    return args


def main(argv=None, default_model=None):
    args = parse_args(argv, default_model)
    device = cuda_device(args.gpu)
    print(f"Selected downstream model: {args.model}", flush=True)
    split_directory = args.split_dir or (args.data_root / "downstream_train")
    cv_folds, run_folds = resolve_fold_plan(args, split_directory)
    if args.smoke_test:
        run_folds = run_folds[:1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sr_model_spec = load_sr_model_spec(args.sr_config) if args.resume is None else None
    sr_config_sha256 = file_sha256(args.sr_config) if args.sr_config is not None else None
    fold_results = []
    for fold in run_folds:
        seed_everything(args.seed + fold)
        train_data, val_data = load_fold_datasets(
            args.data_root, fold, args.labels_csv, args.split_dir, augmentation=args.augmentation)
        provenance = data_provenance(args.data_root, train_data)
        for name, dataset in (("train", train_data), ("val", val_data)):
            counts = [sample["n_regions"] for sample in dataset.samples]
            if counts:
                print(f"fold={fold} {name}: {len(dataset)} WSIs; regions={sum(counts)}; "
                      f"N min/mean/max={min(counts)}/{np.mean(counts):.3f}/{max(counts)}", flush=True)
            else:
                print(f"fold={fold} {name}: 0 WSIs", flush=True)
            if not args.smoke_test and {sample["label"] for sample in dataset.samples} != {0, 1}:
                raise ValueError(f"fold {fold} {name} requires both Normal and Tumor labels")
        print("Labels={}; fold CSV={}; manifest split/slide_manifest ignored".format(
            train_data.labels_csv, train_data.split_csv), flush=True)
        fold_dir = args.output_dir / f"fold_{fold}"
        if not args.smoke_test and args.resume is None and fold_dir.exists() and any(fold_dir.iterdir()):
            raise FileExistsError(f"Output already exists: {fold_dir}; use --resume or a new --output-dir")

        resumed = None
        if args.resume is not None:
            resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
            if (resumed.get("format") != CHECKPOINT_FORMAT
                    or checkpoint_model_name(resumed) != args.model or resumed["fold"] != fold):
                raise ValueError("Resume requires a complete checkpoint for this model and fold")
            if provenance_hashes(resumed["data_provenance"]) != provenance_hashes(provenance):
                raise ValueError("Manifest/labels/fold CSV changed since the checkpoint")
            for key in ("model", "lambda_sr", "cls_micro_batch", "sr_micro_batch", "lr",
                        "weight_decay", "early_stopping_patience", "seed", "num_folds",
                        "best_metric"):
                saved_value = resumed["config"].get(key, checkpoint_model_name(resumed) if key == "model" else None)
                if saved_value != getattr(args, key):
                    raise ValueError(f"Resume must retain checkpoint setting {key}={saved_value}")
            if args.resume.resolve() != (fold_dir / "last.pth").resolve() or not (fold_dir / "best.pth").is_file():
                raise ValueError("Resume from this output fold's last.pth with its existing best.pth")
            initialization = resumed["experiment"].get("sr_initialization")
            if not initialization or initialization.get("kind") != "scratch":
                raise ValueError("Resume checkpoint was not created by scratch SR initialization")
            saved_config_sha256 = initialization.get("config_sha256")
            if sr_config_sha256 is not None and sr_config_sha256 != saved_config_sha256:
                raise ValueError("--sr-config differs from the configuration stored by the resume checkpoint")
            experiment = experiment_signature(args, provenance, train_data.split_csv.parent,
                                              cv_folds, saved_config_sha256)
            best = torch.load(fold_dir / "best.pth", map_location="cpu", weights_only=False)
            with (fold_dir / "history.csv").open(newline="", encoding="utf-8") as handle:
                history_rows = list(csv.DictReader(handle))
            validate_resume_state(resumed, best, history_rows, fold, args.epochs, provenance, experiment)
            del best, history_rows
            model = build_from_spec(args.model, args.sr_root, resumed["model_spec"], device)
            model.load_state_dict(resumed["model_state"], strict=True)
        else:
            experiment = experiment_signature(args, provenance, train_data.split_csv.parent,
                                              cv_folds, sr_config_sha256)
            # Re-seed immediately before construction for reproducible initialization.
            seed_everything(args.seed + fold)
            model = build_scratch_model(args.model, args.sr_root, sr_model_spec, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        start_epoch = 1
        best_score, best_epoch = (-float("inf") if args.best_metric == "auc" else float("inf")), 0
        early_stopping_best_auc, early_stopping_bad_epochs = -float("inf"), 0
        if resumed is not None:
            optimizer.load_state_dict(resumed["optimizer_state"])
            start_epoch = resumed["epoch"] + 1
            best_score = resumed["metrics"]["best_score"]
            best_epoch = resumed["metrics"]["best_epoch"]
            early_stopping_best_auc = resumed["metrics"].get(
                "early_stopping_best_auc", resumed["metrics"].get("best_auc", -float("inf")))
            early_stopping_bad_epochs = resumed["metrics"].get("early_stopping_bad_epochs", 0)
            rng = resumed["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state(rng["cuda"], device=device)
            del resumed, rng
        else:
            seed_everything(args.seed + fold + 100000)
        torch.cuda.empty_cache()

        if args.smoke_test:
            result = train_wsi(model, train_data[0], optimizer, device,
                               args.cls_micro_batch, args.sr_micro_batch, args.lambda_sr, verbose=True)
            result["fold"] = fold
            result["data_provenance"] = provenance
            result["experiment"] = experiment
            result["config"] = serialized_config(args)
            report = args.output_dir / f"smoke_{args.model}_fold_{fold}_cls{args.cls_micro_batch}_sr{args.sr_micro_batch}.json"
            report.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            print(f"PASS: one complete real WSI forward/backward/step; report={report}", flush=True)
            return

        fold_dir.mkdir(parents=True, exist_ok=True)
        loader_args = dict(batch_size=1, num_workers=args.num_workers, collate_fn=collate_one_wsi)
        train_loader = DataLoader(train_data, shuffle=True, **loader_args)
        # Keep the extra offline pass from advancing the global training RNG.
        train_eval_loader = DataLoader(
            train_data.evaluation_view(),
            shuffle=False,
            generator=torch.Generator().manual_seed(args.seed + fold),
            **loader_args,
        )
        val_loader = DataLoader(val_data, shuffle=False, **loader_args)
        completed_epoch = start_epoch - 1
        stopped_early = False
        history_path = fold_dir / "history.csv"
        with history_path.open("a" if args.resume else "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            if not args.resume:
                writer.writeheader()
            for epoch in range(start_epoch, args.epochs + 1):
                sums = {key: 0.0 for key in ("loss_cls", "loss_sr", "loss_total")}
                train_labels, train_preds = [], []
                for index, sample in enumerate(train_loader, start=1):
                    fold_wsi_iteration = (epoch - 1) * len(train_data) + index
                    gradient_log_context = None
                    if fold_wsi_iteration % GRADIENT_LOG_INTERVAL == 0:
                        gradient_log_context = {
                            "fold": fold, "epoch": epoch,
                            "fold_wsi_iteration": fold_wsi_iteration,
                            "epoch_wsi": f"{index}/{len(train_data)}",
                        }
                    result = train_wsi(model, sample, optimizer, device,
                                       args.cls_micro_batch, args.sr_micro_batch, args.lambda_sr,
                                       gradient_log_context=gradient_log_context)
                    for key in sums:
                        sums[key] += result[key]
                    train_labels.append(result["label"])
                    train_preds.append(result["prediction"])
                    print("model={} fold={} epoch={}/{} iteration={} WSI={}/{} {} N={} loss={:.6f} "
                          "seconds={:.2f} peak={:.3f} GiB".format(
                              args.model, fold, epoch, args.epochs, fold_wsi_iteration,
                              index, len(train_data),
                              sample["slide_id"], sample["n_regions"], result["loss_total"],
                              result["seconds"], result["max_memory_allocated_gb"]),
                          flush=True)
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                validation = evaluate(model, val_loader, device, args.cls_micro_batch)
                predictions = validation.pop("predictions")
                train_evaluation = evaluate(model, train_eval_loader, device, args.cls_micro_batch)
                train_evaluation.pop("predictions")
                train_means = {key: value / len(train_data) for key, value in sums.items()}
                train_counts = binary_class_counts(train_labels, train_preds)
                val_labels = [prediction["label"] for prediction in predictions]
                val_preds = [prediction["prediction"] for prediction in predictions]
                val_counts = binary_class_counts(val_labels, val_preds)
                row = {
                    "epoch": epoch,
                    "train_loss_cls": train_means["loss_cls"],
                    "train_loss_sr": train_means["loss_sr"],
                    "train_loss_total": train_means["loss_total"],
                    "train_auc": train_evaluation["auc"],
                    "train_acc": train_evaluation["acc"],
                    "train_bacc": train_evaluation["bacc"],
                    "val_loss": validation["loss"],
                    "val_auc": validation["auc"],
                    "val_acc": validation["acc"],
                    "val_bacc": validation["bacc"],
                }
                current_score = validation_metric(validation, args.best_metric)
                improved = is_better(args.best_metric, current_score, best_score)
                if improved:
                    best_score, best_epoch = current_score, epoch
                early_stopping_best_auc, early_stopping_bad_epochs, should_stop = early_stopping_step(
                    validation["auc"], early_stopping_best_auc,
                    early_stopping_bad_epochs, args.early_stopping_patience)
                metrics = dict(validation, best_metric=args.best_metric, best_score=best_score,
                               best_epoch=best_epoch, best_auc=validation["auc"],
                               early_stopping_best_auc=early_stopping_best_auc,
                               early_stopping_bad_epochs=early_stopping_bad_epochs,
                               early_stopped=should_stop)
                if improved:
                    save_checkpoint(fold_dir / "best.pth", model, optimizer, fold, epoch, metrics, args, provenance, experiment)
                    with (fold_dir / "best_val_predictions.csv").open("w", newline="", encoding="utf-8") as f:
                        prediction_writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
                        prediction_writer.writeheader()
                        prediction_writer.writerows(predictions)
                save_checkpoint(fold_dir / "last.pth", model, optimizer, fold, epoch, metrics, args, provenance, experiment)
                writer.writerow(row)
                handle.flush()
                print("Epoch {:02d} | model={} fold={} | train_cls={:.6f} train_sr={:.6f} "
                      "train_total={:.6f} train_auc={:.6f} train_acc={:.6f} train_bacc={:.6f} | "
                      "val_loss={:.6f} val_auc={:.6f} val_acc={:.6f} val_bacc={:.6f} | "
                      "best {}={:.6f} at epoch={}".format(
                          epoch, args.model, fold, row["train_loss_cls"], row["train_loss_sr"],
                          row["train_loss_total"], row["train_auc"], row["train_acc"],
                          row["train_bacc"], row["val_loss"], row["val_auc"], row["val_acc"],
                          row["val_bacc"], args.best_metric, best_score, best_epoch), flush=True)
                print("Train: GT Normal={} Tumor={} | Pred Normal={} Tumor={} | "
                       "Val: GT Normal={} Tumor={} | Pred Normal={} Tumor={}".format(
                          train_counts["gt_normal"], train_counts["gt_tumor"],
                          train_counts["pred_normal"], train_counts["pred_tumor"],
                          val_counts["gt_normal"], val_counts["gt_tumor"],
                          val_counts["pred_normal"], val_counts["pred_tumor"]), flush=True)
                print("Early stopping: val_auc best={:.6f}; no-improvement epochs={}/{}".format(
                    early_stopping_best_auc, early_stopping_bad_epochs,
                    args.early_stopping_patience), flush=True)
                completed_epoch = epoch
                if should_stop:
                    stopped_early = True
                    print("EARLY STOP model={} fold={} at epoch={}: val AUC did not improve "
                          "for {} consecutive epochs.".format(
                              args.model, fold, epoch, args.early_stopping_patience), flush=True)
                    break
        best = torch.load(fold_dir / "best.pth", map_location="cpu", weights_only=False)
        fold_metrics = {key: best["metrics"][key] for key in ("auc", "acc", "bacc", "loss")}
        fold_metrics["val_loss"] = fold_metrics.pop("loss")
        fold_metrics.update(model_name=args.model, fold=fold, best_epoch=best["epoch"],
                            best_metric=args.best_metric, best_score=best["metrics"]["best_score"],
                            epochs_completed=completed_epoch, max_epochs=args.epochs,
                            early_stopped=stopped_early, experiment=experiment)
        (fold_dir / "metrics.json").write_text(json.dumps(fold_metrics, indent=2), encoding="utf-8")
        with history_path.open(newline="", encoding="utf-8") as handle:
            history_rows = list(csv.DictReader(handle))
        (fold_dir / "history.json").write_text(json.dumps(history_rows, indent=2), encoding="utf-8")
        fold_results.append(fold_metrics)
        del best, optimizer, model, train_loader, train_eval_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    metric_paths = [args.output_dir / "fold_{}".format(fold) / "metrics.json" for fold in cv_folds]
    if all(path.is_file() for path in metric_paths):
        fold_results = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
        summary = combine_fold_metrics(fold_results, cv_folds)
        (args.output_dir / "summary.json").write_text(
            json.dumps({"model_name": args.model, "folds": fold_results, "mean_std": summary,
                        "std_ddof": 1, "num_folds": len(cv_folds), "best_metric": args.best_metric},
                       indent=2), encoding="utf-8")
        with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=("fold", "best_epoch", "auc", "acc", "bacc", "val_loss"))
            writer.writeheader()
            writer.writerows({key: row.get(key, row.get("loss")) for key in writer.fieldnames}
                             for row in fold_results)
        for key, values in summary.items():
            print("{} {}-fold {}: {:.6f} +/- {:.6f} (sample std, ddof=1)".format(
                args.model, len(cv_folds), key, values["mean"], values["std"]))
        try:
            plots = plot_run_directory(args.output_dir, cv_folds)
            history_json = export_history_json(args.output_dir, cv_folds)
            print("Training curves: {}; combined history: {}".format(plots, history_json), flush=True)
        except RuntimeError as exc:
            print("Plot skipped: {}".format(exc), flush=True)
    else:
        print("Completed folds in this run: {}; summary waits for all {} folds.".format(
            [row["fold"] for row in fold_results], len(cv_folds)))


if __name__ == "__main__":
    main()
