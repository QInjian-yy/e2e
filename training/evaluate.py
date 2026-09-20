"""Classification-only evaluation for Mean-ResNet checkpoints."""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from downstream.registry import available_models, build_from_spec, checkpoint_model_name
from training.engine import evaluate
from training.train import CHECKPOINT_FORMAT, cuda_device, data_provenance, provenance_hashes
from wsi_data import DEFAULT_DATA_ROOT, collate_one_wsi, load_fold_datasets


def main(argv=None, expected_model=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=available_models(),
                        help="Optional assertion; otherwise inferred from the checkpoint")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--sr-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Complete E2E fold best.pth")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cls-micro-batch", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--predictions-csv", type=Path)
    args = parser.parse_args(argv)
    device = cuda_device(args.gpu)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Expected checkpoint format {CHECKPOINT_FORMAT}")
    model_name = checkpoint_model_name(saved)
    required_model = expected_model or args.model
    if required_model is not None and model_name != required_model:
        raise ValueError(f"Checkpoint model is {model_name}, not {required_model}")
    _, val_data = load_fold_datasets(args.data_root, saved["fold"], args.labels_csv, args.split_dir)
    if provenance_hashes(data_provenance(args.data_root, val_data)) != provenance_hashes(saved["data_provenance"]):
        raise ValueError("Manifest/labels/fold CSV do not match this E2E checkpoint")
    micro_batch = (saved["config"]["cls_micro_batch"] if args.cls_micro_batch is None
                   else args.cls_micro_batch)
    if micro_batch < 1 or args.num_workers < 0:
        parser.error("cls-micro-batch must be positive and num-workers nonnegative")
    model = build_from_spec(model_name, args.sr_root, saved["model_spec"], device)
    model.load_state_dict(saved["model_state"], strict=True)
    fold, epoch = saved["fold"], saved["epoch"]
    del saved
    loader = DataLoader(val_data, batch_size=1, num_workers=args.num_workers,
                        collate_fn=collate_one_wsi)
    result = evaluate(model, loader, device, micro_batch)
    predictions = result.pop("predictions")
    print(f"model={model_name}; fold={fold}; best epoch={epoch}; "
          f"classification-only validation={result}", flush=True)
    if args.predictions_csv:
        args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)


if __name__ == "__main__":
    main()
