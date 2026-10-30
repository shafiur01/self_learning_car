"""Freeze development data, run paired supervised trials, and record all results.

Run from the repository root: python -m research.experiments.run
This command never evaluates test clips or replaces an application model.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gesture_commands.data import (
    CONVENTION, FEATURES, FRAMES, LABELS, SECONDS, VERSION,
    features, load_clip, quality,
)
from gesture_commands.model import GestureNet
from gesture_commands.train import require_classes, tensors
from .augment import VARIANTS, augment

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "001_augmentation.json"
DEFAULT_OUTPUT = Path(__file__).parent / "runs"
SOURCE_FILES = (
    "research/experiments/run.py", "research/experiments/augment.py",
    "gesture_commands/data.py", "gesture_commands/model.py", "gesture_commands/train.py",
    "requirements.txt", "gesture_commands/requirements.txt",
    "research/PROTOCOL.md",
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    if config.get("experiment_id") != "001_augmentation":
        raise ValueError("This runner implements experiment 001_augmentation only.")
    for key in ("epochs", "batch_size", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    seeds = config["seeds"]
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Use distinct nonnegative integer seeds.")
    variants = config["variants"]
    if not variants or variants[0] != "baseline" or len(set(variants)) != len(variants):
        raise ValueError("Variants must start with baseline and contain no duplicates.")
    if any(variant not in VARIANTS for variant in variants):
        raise ValueError("Unknown variant.")
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    settings = config["augmentation"]
    if not math.isfinite(settings["jitter_std"]) or not 0 <= settings["jitter_std"] <= 0.1:
        raise ValueError("jitter_std must be between 0 and 0.1.")
    for key in ("jitter_anchors", "time_segments"):
        if type(settings[key]) is not int or not 2 <= settings[key] < FRAMES:
            raise ValueError(f"{key} must be an integer between 2 and 31.")
    low, high = settings["time_increment_range"]
    if not math.isfinite(low + high) or not 0 < low <= high:
        raise ValueError("Time increments must be positive, ordered, and finite.")


def development_snapshot(folder):
    """Only train/validation features enter this study; all sessions stay intact.

Test metadata is inspected solely to check session separation. Test coordinates,
labels, and predictions do not enter the experiment manifest or the trainer.
"""
    dataset = {"train": [], "validation": []}
    records, sessions, content_splits = [], {}, {}
    for path in sorted(Path(folder).glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
        split, session = metadata["split"], metadata["session"]
        if split not in ("train", "validation", "test") or not isinstance(session, str) or not session:
            raise ValueError(f"Invalid split/session: {path}")
        if session in sessions and sessions[session] != split:
            raise ValueError(f"Session {session} appears in multiple splits.")
        sessions[session] = split
        if split == "test":
            continue
        before = sha256(path)
        xyz, times, visible, meta = load_clip(path)
        if sha256(path) != before:
            raise ValueError(f"Recording changed while it was read: {path}")
        # Catch exact raw development duplicates even when their labels differ.
        identity = hashlib.sha256()
        for array in (xyz, times - times[0], visible, np.asarray([meta["aspect"]], dtype=np.float64)):
            identity.update(array.tobytes())
        content_id = identity.hexdigest()
        if content_id in content_splits and content_splits[content_id] != split:
            raise ValueError(f"An identical raw clip appears in train and validation: {path}")
        content_splits[content_id] = split
        name = os.path.relpath(path.resolve(), PROJECT)
        dataset[split].append((features(xyz, times, visible, meta["aspect"]),
                               LABELS.index(meta["label"]), name))
        records.append({"file": name, "sha256": before, "raw_content_sha256": content_id,
                        "split": split, "session": session, "label": meta["label"],
                        "source": meta.get("source", "unknown"),
                        "passes_tracking_quality": quality(times, visible)})
    for split, minimum in (("train", 5), ("validation", 2)):
        require_classes(dataset[split], split, minimum)
    summary = {}
    for split in dataset:
        items = [record for record in records if record["split"] == split]
        summary[split] = {
            "clips": len(items), "sessions": len({record["session"] for record in items}),
            "labels": dict(Counter(record["label"] for record in items)),
            "sources": dict(Counter(record["source"] for record in items)),
            "clips_failing_movement_tracking_quality": sum(not record["passes_tracking_quality"] for record in items),
        }
    manifest = {"schema_version": VERSION, "convention": CONVENTION,
                "labels": list(LABELS), "frames": FRAMES, "features": FEATURES,
                "summary": summary, "records": records,
                "scope": "train and validation only; test data excluded"}
    return dataset, manifest


def validation_metrics(probabilities, targets):
    predicted = probabilities.argmax(1)
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(matrix, (targets, predicted), 1)
    true_positive = np.diag(matrix)
    precision = np.divide(true_positive, matrix.sum(0), out=np.zeros(len(LABELS)), where=matrix.sum(0) != 0)
    recall = np.divide(true_positive, matrix.sum(1), out=np.zeros(len(LABELS)), where=matrix.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(LABELS)), where=(precision + recall) != 0)
    movement = predicted < LABELS.index("STOP")
    no_command = targets == LABELS.index("NO_COMMAND")
    stop = targets == LABELS.index("STOP")
    return {"accuracy": float(np.mean(predicted == targets)), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall.mean()),
            "nll": float(-np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-12)).mean()),
            "no_command_to_motion_count": int(np.sum(no_command & movement)),
            "no_command_to_motion_rate": float(np.sum(no_command & movement) / max(1, no_command.sum())),
            "stop_to_motion_count": int(np.sum(stop & movement)),
            "stop_recall": float(recall[LABELS.index("STOP")]),
            "clips": len(targets), "labels": list(LABELS),
            "confusion_matrix_actual_rows_predicted_columns": matrix.tolist(),
            "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                   "f1": float(f1[i]), "support": int(matrix[i].sum())}
                          for i, label in enumerate(LABELS)}}


def train_trial(dataset, config, variant, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    # Augmentation randomness is separate: paired trials have identical dropout
    # random streams and batch ordering, despite sampling extra transformations.
    augmentation_rng = torch.Generator().manual_seed(seed + 1_000_000)
    model = GestureNet()
    initial_sha = hashlib.sha256(b"".join(value.numpy().tobytes() for value in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    counts = require_classes(dataset["train"], "train", 5)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(LABELS) * counts), dtype=torch.float32))
    x, y = tensors(dataset["train"])
    vx, vy = tensors(dataset["validation"])
    best_loss, best_epoch, best_state = float("inf"), None, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for indices in torch.randperm(len(x)).split(config["batch_size"]):
            batch = augment(x[indices], variant, augmentation_rng, config["augmentation"])
            optimizer.zero_grad()
            loss = loss_fn(model(batch), y[indices])
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss in {variant}, seed {seed}, epoch {epoch}.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        model.eval()
        with torch.inference_mode():
            val_loss = nn.functional.cross_entropy(model(vx), vy).item()
        if not math.isfinite(val_loss):
            raise ValueError("Non-finite validation loss.")
        history.append({"epoch": epoch, "weighted_train_loss_batch_mean": total_loss / len(x),
                        "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise ValueError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    with torch.inference_mode():
        probabilities = model(vx).softmax(1).numpy()
    metrics = validation_metrics(probabilities, vy.numpy())
    metrics.update(variant=variant, seed=seed, best_epoch=best_epoch,
                   training_seconds=training_seconds, initial_weights_sha256=initial_sha,
                   parameters=sum(parameter.numel() for parameter in model.parameters()))
    return model, metrics, history, probabilities


def summarize(results, config):
    fields = ("accuracy", "macro_f1", "balanced_accuracy", "nll",
              "no_command_to_motion_rate", "stop_recall", "training_seconds")
    variants = {}
    for variant in config["variants"]:
        trials = [result for result in results if result["variant"] == variant]
        variants[variant] = {field: {
            "mean": float(np.mean([result[field] for result in trials])),
            "std_across_seeds": float(np.std([result[field] for result in trials], ddof=1)) if len(trials) > 1 else None,
            "min": float(min(result[field] for result in trials)),
            "max": float(max(result[field] for result in trials)),
        } for field in fields}
    baseline = {result["seed"]: result for result in results if result["variant"] == "baseline"}
    paired = {}
    for variant in config["variants"][1:]:
        trials = [result for result in results if result["variant"] == variant]
        if any(result["initial_weights_sha256"] != baseline[result["seed"]]["initial_weights_sha256"] for result in trials):
            raise ValueError("Paired conditions started from different model weights.")
        paired[variant] = {field: {
            "by_seed": {str(result["seed"]): result[field] - baseline[result["seed"]][field] for result in trials},
            "mean_difference": float(np.mean([result[field] - baseline[result["seed"]][field] for result in trials])),
        } for field in ("accuracy", "macro_f1", "no_command_to_motion_rate")}
    return {"evaluation_split": "validation", "variants": variants, "paired_differences": paired,
            "interpretation": "Seed spread measures optimization variability on the same validation clips, not uncertainty across people/sessions. Epoch selection and comparison both use validation. No test result or novelty claim."}


def markdown_report(summary, manifest, config):
    lines = ["# Experiment 001: augmentation pilot", "",
             "**Development findings only. These are validation results, not untouched test results.**", "",
             config["question"], "",
             f"Data: {manifest['summary']['train']['clips']} training clips from {manifest['summary']['train']['sessions']} sessions; "
             f"{manifest['summary']['validation']['clips']} validation clips from {manifest['summary']['validation']['sessions']} sessions.", "",
             f"Seeds: {config['seeds']}; epochs: {config['epochs']}; architecture/optimizer/data fixed.", "",
             "| Condition | Accuracy mean ± seed SD | Macro F1 mean ± seed SD | NO_COMMAND → motion mean | Training seconds mean |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, values in summary["variants"].items():
        accuracy, f1 = values["accuracy"], values["macro_f1"]
        accuracy_sd = "n/a" if accuracy["std_across_seeds"] is None else f"{accuracy['std_across_seeds'] * 100:.2f}"
        f1_sd = "n/a" if f1["std_across_seeds"] is None else f"{f1['std_across_seeds']:.4f}"
        lines.append(f"| {name} | {accuracy['mean'] * 100:.2f}% ± {accuracy_sd} pp | "
                     f"{f1['mean']:.4f} ± {f1_sd} | {values['no_command_to_motion_rate']['mean'] * 100:.2f}% | "
                     f"{values['training_seconds']['mean']:.2f} |")
    lines.extend(["", "## Paired differences from baseline", "",
                  "| Condition | Mean accuracy change | Mean macro F1 change | Seeds with higher accuracy |",
                  "| --- | ---: | ---: | ---: |"])
    for name, values in summary["paired_differences"].items():
        differences = values["accuracy"]["by_seed"]
        lines.append(f"| {name} | {values['accuracy']['mean_difference'] * 100:+.2f} pp | "
                     f"{values['macro_f1']['mean_difference']:+.4f} | {sum(value > 0 for value in differences.values())}/{len(differences)} |")
    lines.extend(["", "## Interpretation limits", "",
                  "- Validation chooses the checkpoint and compares conditions; results have selection bias.",
                  "- Five seeds repeat optimization, not five independent datasets.",
                  "- Raw NO_COMMAND → motion classification rates do not measure live gated command events or false starts per minute.",
                  "- Tracking-quality failures remain in the data, including genuine no-hand NO_COMMAND clips; the manifest records them.",
                  "- Augmentation creates transformed views, not additional independent human recordings.",
                  "- No old or new test clips were evaluated, and no active application checkpoint was replaced.",
                  "- This pilot does not test label-budget reduction, semi-supervision, self-supervision, or novelty.", ""])
    return "\n".join(lines)


def run(config_path, output_root):
    config = json.loads(config_path.read_text())
    validate_config(config)
    folder = Path(config["data"])
    if not folder.is_absolute():
        folder = PROJECT / folder
    dataset, manifest = development_snapshot(folder)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "manifest.json", manifest)
    train_x, train_y = tensors(dataset["train"])
    val_x, val_y = tensors(dataset["validation"])
    np.savez_compressed(output / "development_features.npz", train_x=train_x.numpy(), train_y=train_y.numpy(),
                        validation_x=val_x.numpy(), validation_y=val_y.numpy())
    source_hashes = {}
    for name in SOURCE_FILES:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, destination)
        source_hashes[name] = sha256(destination)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--short"], cwd=PROJECT, capture_output=True, text=True)
    environment = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                   "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
                   "device": "cpu", "threads": config["threads"], "git_commit": commit.stdout.strip(),
                   "git_status_at_start": status.stdout, "source_sha256": source_hashes,
                   "config_sha256": sha256(output / "config.json"), "manifest_sha256": sha256(output / "manifest.json"),
                   "features_sha256": sha256(output / "development_features.npz")}
    write_json(output / "environment.json", environment)
    write_json(output / "status.json", {"state": "running", "completed_trials": 0})
    print(f"Study folder: {output}", flush=True)
    print(json.dumps(manifest["summary"], indent=2), flush=True)
    records = manifest["records"]
    development_sessions = sorted({record["session"] for record in records})
    results = []
    try:
        for seed in config["seeds"]:
            for variant in config["variants"]:
                trial = output / f"{variant}_seed_{seed}"
                trial.mkdir()
                model, metrics, history, probabilities = train_trial(dataset, config, variant, seed)
                write_json(trial / "metrics.json", metrics)
                write_json(trial / "history.json", history)
                write_json(trial / "validation_predictions.json", [
                    {"file": sample[2], "actual": LABELS[sample[1]],
                     "predicted": LABELS[int(probability.argmax())], "scores": probability.tolist()}
                    for sample, probability in zip(dataset["validation"], probabilities)])
                torch.save({"state_dict": model.state_dict(), "schema_version": VERSION,
                            "convention": CONVENTION, "labels": list(LABELS), "frames": FRAMES,
                            "seconds": SECONDS, "seed": seed, "best_epoch": metrics["best_epoch"],
                            "development_sessions": development_sessions,
                            "development_files": [record["file"] for record in records],
                            "study": config["experiment_id"], "variant": variant,
                            "manifest_sha256": environment["manifest_sha256"]}, trial / "model.pt")
                results.append(metrics)
                write_json(output / "trials.json", results)
                write_json(output / "status.json", {"state": "running", "completed_trials": len(results)})
                print(f"{variant:10s} seed={seed}: accuracy={metrics['accuracy']:.2%}, "
                      f"macro F1={metrics['macro_f1']:.4f}, epoch={metrics['best_epoch']}, "
                      f"train={metrics['training_seconds']:.2f}s", flush=True)
        summary = summarize(results, config)
        write_json(output / "summary.json", summary)
        (output / "report.md").write_text(markdown_report(summary, manifest, config))
        write_json(output / "status.json", {"state": "complete", "completed_trials": len(results)})
    except BaseException as error:
        write_json(output / "status.json", {"state": "interrupted_or_failed", "completed_trials": len(results),
                                            "error": f"{type(error).__name__}: {error}"})
        raise
    print(f"Read results: {output / 'report.md'}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        run(args.config.resolve(), args.output.resolve())
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
