"""Plot train/val loss and classification metrics from E2E run directories."""

import argparse
import csv
import json
from pathlib import Path

HISTORY_FIELDS = (
    "epoch", "train_loss_cls", "train_loss_sr", "train_loss_total",
    "train_auc", "train_acc", "train_bacc",
    "val_loss", "val_auc", "val_acc", "val_bacc",
)


def _read_history(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {"epoch": int(row["epoch"])}
            for key in HISTORY_FIELDS[1:]:
                legacy = {"train_loss_cls": "loss_cls", "train_loss_sr": "loss_sr",
                          "train_loss_total": "loss_total"}.get(key, key)
                value = row.get(key, row.get(legacy))
                if value in (None, ""):
                    if key in ("train_auc", "train_acc", "train_bacc"):
                        parsed[key] = float("nan")
                        continue
                    raise ValueError("{} missing column {} in {}".format(path, key, row))
                parsed[key] = float(value)
            rows.append(parsed)
    if not rows:
        raise ValueError("No history rows in {}".format(path))
    return rows


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting; install it or inspect history.csv") from exc
    return plt


def plot_fold_history(history_rows, output_path, fold):
    plt = _require_matplotlib()
    epochs = [row["epoch"] for row in history_rows]
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    figure.suptitle("Fold {} training curves".format(fold))

    axes[0, 0].plot(epochs, [row["train_loss_total"] for row in history_rows], label="train")
    axes[0, 0].plot(epochs, [row["val_loss"] for row in history_rows], label="val")
    axes[0, 0].set_title("Train total / val classification loss")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, [row["train_loss_cls"] for row in history_rows], label="train cls")
    axes[0, 1].plot(epochs, [row["train_loss_sr"] for row in history_rows], label="train sr")
    axes[0, 1].set_title("Train loss components")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, [row["train_auc"] for row in history_rows], label="train")
    axes[1, 0].plot(epochs, [row["val_auc"] for row in history_rows], label="val")
    axes[1, 0].set_title("WSI AUC")
    axes[1, 0].set_ylabel("AUC")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, [row["train_acc"] for row in history_rows], label="train acc")
    axes[1, 1].plot(epochs, [row["train_bacc"] for row in history_rows], label="train bacc")
    axes[1, 1].plot(epochs, [row["val_acc"] for row in history_rows], linestyle="--", label="val acc")
    axes[1, 1].plot(epochs, [row["val_bacc"] for row in history_rows], linestyle="--", label="val bacc")
    axes[1, 1].set_title("WSI accuracy")
    axes[1, 1].set_ylabel("score")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    for axis in axes[1, :]:
        axis.set_xlabel("epoch")

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_summary(output_dir, folds):
    plt = _require_matplotlib()
    histories = {fold: _read_history(output_dir / "fold_{}".format(fold) / "history.csv")
                 for fold in folds}
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    figure.suptitle("Cross-validation summary ({} folds)".format(len(folds)))

    for metric, axis, ylabel in (
        ("val_loss", axes[0], "val loss"),
        ("val_auc", axes[1], "AUC"),
        ("val_acc", axes[2], "ACC"),
    ):
        for fold, rows in histories.items():
            axis.plot([row["epoch"] for row in rows], [row[metric] for row in rows],
                      alpha=0.45, label="fold {}".format(fold))
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        if metric in ("val_auc", "val_acc"):
            axis.set_ylim(0.0, 1.0)
    axes[2].legend(loc="lower right", fontsize=8)

    figure.tight_layout()
    plots = Path(output_dir) / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    figure.savefig(plots / "cv_summary.png", dpi=150)
    plt.close(figure)


def plot_run_directory(output_dir, folds):
    output_dir = Path(output_dir)
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        history_path = output_dir / "fold_{}".format(fold) / "history.csv"
        if not history_path.is_file():
            continue
        plot_fold_history(_read_history(history_path), plots / "fold_{}_curves.png".format(fold), fold)
    if len(folds) > 1 and all((output_dir / "fold_{}".format(fold) / "history.csv").is_file()
                               for fold in folds):
        plot_summary(output_dir, folds)
    return plots


def export_history_json(output_dir, folds):
    combined = {}
    for fold in folds:
        history_path = Path(output_dir) / "fold_{}".format(fold) / "history.csv"
        if history_path.is_file():
            combined["fold_{}".format(fold)] = _read_history(history_path)
    summary_path = Path(output_dir) / "summary.json"
    if summary_path.is_file():
        combined["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    export_path = Path(output_dir) / "training_history.json"
    export_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return export_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="*", default=None,
                        help="Fold ids to plot (default: all fold_* under output-dir)")
    args = parser.parse_args()
    if args.folds is None:
        folds = sorted(int(path.name.split("_")[1])
                       for path in args.output_dir.glob("fold_*/history.csv"))
    else:
        folds = list(args.folds)
    if not folds:
        raise ValueError("No fold history found under {}".format(args.output_dir))
    plots = plot_run_directory(args.output_dir, folds)
    export = export_history_json(args.output_dir, folds)
    print("Wrote plots to {}; history JSON={}".format(plots, export), flush=True)


if __name__ == "__main__":
    main()
