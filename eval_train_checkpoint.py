"""Evaluate each fold's training split with its saved best checkpoint."""

import argparse
import csv
import gc
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from downstream.registry import build_from_spec, checkpoint_model_name
from training.engine import evaluate
from training.train import CHECKPOINT_FORMAT, cuda_device, data_provenance, provenance_hashes
from wsi_data import collate_one_wsi, load_fold_datasets


def discover_folds(run_dir):
    folds = []
    for checkpoint in sorted(run_dir.glob("fold_*/best.pth")):
        suffix = checkpoint.parent.name.removeprefix("fold_")
        if suffix.isdigit():
            folds.append(int(suffix))
    if not folds:
        raise FileNotFoundError(f"No fold_*/best.pth found in {run_dir}")
    return folds


def configured_path(config, name):
    value = config.get(name)
    return None if value in (None, "") else Path(value)


def write_predictions(path, predictions):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("slide_id", "label", "prob", "pred"))
        writer.writeheader()
        for prediction in predictions:
            writer.writerow({
                "slide_id": prediction["slide_id"],
                "label": prediction["label"],
                "prob": prediction["p_tumor"],
                "pred": prediction["prediction"],
            })
    temporary.replace(path)


def evaluate_fold(run_dir, fold, gpu_override=None):
    checkpoint_path = run_dir / f"fold_{fold}" / "best.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{checkpoint_path} is not a {CHECKPOINT_FORMAT} checkpoint")
    if saved.get("fold") != fold:
        raise ValueError(f"{checkpoint_path} stores fold {saved.get('fold')}, expected {fold}")

    config = saved["config"]
    data_root = Path(config["data_root"])
    sr_root = Path(config["sr_root"])
    labels_csv = configured_path(config, "labels_csv")
    split_dir = configured_path(config, "split_dir")
    train_data, _ = load_fold_datasets(data_root, fold, labels_csv, split_dir)

    current_provenance = data_provenance(data_root, train_data)
    if provenance_hashes(current_provenance) != provenance_hashes(saved["data_provenance"]):
        raise ValueError("Manifest/labels/fold CSV do not match this checkpoint")

    gpu = int(config.get("gpu", 0) if gpu_override is None else gpu_override)
    device = cuda_device(gpu)
    model_name = checkpoint_model_name(saved)
    downstream_spec = saved.get("downstream_spec")
    if downstream_spec is None:
        model = build_from_spec(model_name, sr_root, saved["model_spec"], device)
    else:
        model = build_from_spec(model_name, sr_root, saved["model_spec"], device,
                                downstream_spec)
    model.load_state_dict(saved["model_state"], strict=True)
    model.eval()

    loader = DataLoader(
        train_data,
        batch_size=1,
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collate_one_wsi,
    )
    with torch.no_grad():
        result = evaluate(model, loader, device, int(config["cls_micro_batch"]))
    predictions = result.pop("predictions")

    output_path = checkpoint_path.parent / "train_eval_predictions.csv"
    write_predictions(output_path, predictions)
    print(
        "fold={} train loss={:.6f} AUC={:.6f} ACC={:.6f} BACC={:.6f} predictions={}".format(
            fold, result["loss"], result["auc"], result["acc"], result["bacc"], output_path
        ),
        flush=True,
    )

    del loader, model, saved
    gc.collect()
    torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", "--run-dir", dest="run_dir", type=Path, required=True)
    parser.add_argument("--fold", required=True, help="Fold id or 'all'")
    parser.add_argument("--gpu", type=int, help="Override the GPU saved in the checkpoint")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"--run_dir is not a directory: {run_dir}")

    available = discover_folds(run_dir)
    if args.fold == "all":
        folds = available
    else:
        try:
            fold = int(args.fold)
        except ValueError:
            parser.error("--fold must be a nonnegative integer or 'all'")
        if fold < 0 or fold not in available:
            parser.error(f"--fold must be one of {available} or 'all'")
        folds = [fold]

    for fold in folds:
        evaluate_fold(run_dir, fold, args.gpu)


if __name__ == "__main__":
    main()
