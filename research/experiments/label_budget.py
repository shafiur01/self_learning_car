"""Supervised label-budget pilot with fixed training updates and validation checks.

Run: python -m research.experiments.label_budget
Existing labels are withheld to simulate small training budgets. This does not
measure actual annotation savings, evaluate test clips, or activate a model.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gesture_commands.data import CONVENTION, FRAMES, LABELS, SECONDS, VERSION
from gesture_commands.model import GestureNet
from gesture_commands.train import require_classes, tensors
from .run import (DEFAULT_OUTPUT, PROJECT, SOURCE_FILES, development_snapshot,
                  sha256, validation_metrics, write_json)

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "002_label_budget.json"


def validate_config(config):
    if config.get("experiment_id") != "002_label_budget":
        raise ValueError("This runner implements experiment 002_label_budget only.")
    for key in ("optimizer_steps", "validation_every_steps", "batch_size", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["validation_every_steps"] > config["optimizer_steps"]:
        raise ValueError("The validation interval cannot exceed the update budget.")
    budgets = config["labels_per_class"]
    if (not budgets or any(type(n) is not int or n < 5 for n in budgets)
            or budgets != sorted(set(budgets))):
        raise ValueError("Use increasing distinct integer budgets of at least five labels/class.")
    seeds = config["seeds"]
    if (not seeds or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Use distinct nonnegative integer seeds.")
    if type(config["include_all_original_reference"]) is not bool:
        raise ValueError("include_all_original_reference must be boolean.")
    if config["eligible_source"] != "recording":
        raise ValueError("The primary pilot uses original recordings, excluding model-selected corrections.")
    if config["sampling"] != "uniform_with_replacement" or config["augmentation"] != "none":
        raise ValueError("Use the prespecified replacement sampler with no augmentation.")
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive and finite.")


def original_pool(dataset, manifest):
    """Keep corrections frozen in the archive but out of the primary training pool."""
    records = {r["file"]: r for r in manifest["records"] if r["split"] == "train"}
    pool = [s for s in dataset["train"] if records[s[2]]["source"] == "recording"]
    require_classes(pool, "original training", 5)
    seen = set()
    for sample in pool:
        identity = records[sample[2]]["raw_content_sha256"]
        if identity in seen:
            raise ValueError("Duplicate original training contents would inflate the label budget.")
        seen.add(identity)
    return pool


def nested_subsets(pool, manifest, config, seed):
    """Seeded, session-balanced prefixes: five/class is contained in ten/class.

Labels are inspected to simulate balanced budgets; this is not a real label
acquisition algorithm. No training-session recording moves into evaluation.
"""
    records = {r["file"]: r for r in manifest["records"]}
    rng = np.random.default_rng(seed + 2_000_000)
    ordered = []
    for label_index, label in enumerate(LABELS):
        groups = defaultdict(list)
        for sample in sorted(pool, key=lambda s: s[2]):
            if sample[1] == label_index:
                groups[records[sample[2]]["session"]].append(sample)
        if sum(map(len, groups.values())) < max(config["labels_per_class"]):
            raise ValueError(f"Not enough original {label} clips for the largest label budget.")
        sessions = sorted(groups)
        sessions = [sessions[i] for i in rng.permutation(len(sessions))]
        for session in sessions:
            items = groups[session]
            groups[session] = [items[i] for i in rng.permutation(len(items))]
        label_order = []
        offset = 0
        while any(offset < len(groups[s]) for s in sessions):
            for session in sessions:
                if offset < len(groups[session]):
                    label_order.append(groups[session][offset])
            offset += 1
        ordered.append(label_order)
    conditions = {f"labels_{n}_per_class": [s for group in ordered for s in group[:n]]
                  for n in config["labels_per_class"]}
    if config["include_all_original_reference"]:
        conditions["all_original"] = [s for group in ordered for s in group]
    return conditions


def train_steps(train, validation, config, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    batch_rng = torch.Generator().manual_seed(seed + 1_000_000)
    model = GestureNet()
    initial_hash = hashlib.sha256(b"".join(v.numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    counts = require_classes(train, "selected training", 5)
    weights = torch.tensor(counts.sum() / (len(LABELS) * counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    x, y = tensors(train)
    vx, vy = tensors(validation)
    best_loss, best_step, best_state = float("inf"), None, None
    history, loss_sum, interval_steps = [], 0.0, 0
    started = time.perf_counter()
    for step in range(1, config["optimizer_steps"] + 1):
        model.train()
        indices = torch.randint(len(x), (config["batch_size"],), generator=batch_rng)
        optimizer.zero_grad()
        loss = loss_fn(model(x[indices]), y[indices])
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite training loss at step {step}.")
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        interval_steps += 1
        if step % config["validation_every_steps"] == 0 or step == config["optimizer_steps"]:
            model.eval()
            with torch.inference_mode():
                val_loss = nn.functional.cross_entropy(model(vx), vy).item()
            if not math.isfinite(val_loss):
                raise ValueError("Non-finite validation loss.")
            history.append({"step": step, "interval_mean_weighted_train_loss": loss_sum / interval_steps,
                            "validation_loss": val_loss})
            loss_sum, interval_steps = 0.0, 0
            if val_loss < best_loss:
                best_loss, best_step, best_state = val_loss, step, copy.deepcopy(model.state_dict())
    seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        probabilities = model(vx).softmax(1).numpy()
    metrics = validation_metrics(probabilities, vy.numpy())
    metrics.update(seed=seed, best_step=best_step, optimizer_steps=config["optimizer_steps"],
                   validation_checks=len(history), training_seconds=seconds,
                   initial_weights_sha256=initial_hash, parameters=sum(p.numel() for p in model.parameters()),
                   training_clips=len(train), training_labels_by_class=dict(zip(LABELS, map(int, counts))),
                   example_draws=config["optimizer_steps"] * config["batch_size"])
    return model, metrics, history, probabilities


def session_metrics(probabilities, validation, manifest):
    sessions = {r["file"]: r["session"] for r in manifest["records"] if r["split"] == "validation"}
    result = {}
    for session in sorted(set(sessions.values())):
        indices = [i for i, s in enumerate(validation) if sessions[s[2]] == session]
        targets = np.asarray([validation[i][1] for i in indices])
        metrics = validation_metrics(probabilities[indices], targets)
        if not metrics["per_class"]["NO_COMMAND"]["support"]:
            metrics["no_command_to_motion_rate"] = None
        if not metrics["per_class"]["STOP"]["support"]:
            metrics["stop_recall"] = None
        result[session] = metrics
    return result


def summarize(results):
    fields = ("accuracy", "macro_f1", "nll", "no_command_to_motion_rate", "stop_recall", "training_seconds")
    summary = {"evaluation_split": "validation", "conditions": {}, "paired_differences_from_all_original": {}}
    for name in dict.fromkeys(r["condition"] for r in results):
        trials = [r for r in results if r["condition"] == name]
        entry = {"training_clips": trials[0]["training_clips"]}
        for field in fields:
            values = [t[field] for t in trials]
            entry[field] = {"mean": float(np.mean(values)),
                            "std_across_subset_and_model_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        summary["conditions"][name] = entry
    reference = {r["seed"]: r for r in results if r["condition"] == "all_original"}
    for result in results:
        if reference and result["initial_weights_sha256"] != reference[result["seed"]]["initial_weights_sha256"]:
            raise ValueError("Paired label budgets started from different weights.")
    if reference:
        for name in summary["conditions"]:
            if name == "all_original":
                continue
            trials = [r for r in results if r["condition"] == name]
            summary["paired_differences_from_all_original"][name] = {
                field: {"by_seed": {str(t["seed"]): t[field] - reference[t["seed"]][field] for t in trials},
                        "mean_difference": float(np.mean([t[field] - reference[t["seed"]][field] for t in trials]))}
                for field in ("accuracy", "macro_f1", "no_command_to_motion_rate")}
    summary["interpretation"] = (
        "Simulated withholding of already acquired labels; not measured annotation savings. "
        "Seed spread combines subset and optimization variation on the same validation clips. "
        "Validation selects checkpoints and compares budgets; no untouched test, robotics, or unseen-user result.")
    return summary


def markdown_report(summary, manifest, pool, config):
    lines = ["# Experiment 002: supervised label-budget pilot", "",
             "**Development validation only; no measured annotation savings or robotics result.**", "",
             config["question"], "",
             f"Eligible pool: {len(pool)} original recordings; "
             f"{manifest['summary']['train']['clips'] - len(pool)} corrections excluded from primary training.",
             f"Validation: {manifest['summary']['validation']['clips']} clips in "
             f"{manifest['summary']['validation']['sessions']} sessions; seeds {config['seeds']}.",
             f"Every trial: {config['optimizer_steps']} Adam updates, batch size {config['batch_size']}, "
             f"validation every {config['validation_every_steps']} updates; uniform replacement sampling; no augmentation.", "",
             "| Condition | Training labels | Accuracy mean ± run SD | Macro F1 mean | NO_COMMAND → motion mean | Training seconds mean |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, values in summary["conditions"].items():
        sd = values["accuracy"]["std_across_subset_and_model_seeds"]
        spread = "n/a" if sd is None else f"{100 * sd:.2f} pp"
        lines.append(f"| {name} | {values['training_clips']} | {100 * values['accuracy']['mean']:.2f}% ± {spread} | "
                     f"{values['macro_f1']['mean']:.4f} | {100 * values['no_command_to_motion_rate']['mean']:.2f}% | "
                     f"{values['training_seconds']['mean']:.2f} |")
    lines += ["", "## All paired runs", "",
              "| Seed | Condition | Accuracy | Macro F1 | NO_COMMAND → motion count | Best update |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for result in summary["trials"]:
        lines.append(f"| {result['seed']} | {result['condition']} | {result['accuracy']:.2%} | "
                     f"{result['macro_f1']:.4f} | {result['no_command_to_motion_count']} | {result['best_step']} |")
    lines += ["", "## Interpretation", "",
              "- Smaller class-balanced sets are nested within larger sets for each seed; the all-original reference is naturally imbalanced.",
              "- Labels were already acquired. Stratifying by their known labels is a simulation, not a deployable acquisition policy.",
              "- Corrections were selected using earlier models; their acquisition/training history is not a free low-label resource.",
              "- Full batches may repeat a recording. More draws are not more independent observations.",
              "- Run SD combines subset selection and model randomness; it is not uncertainty over new people or sessions.",
              "- Fixed-step replacement sampling and checkpoint checks differ from experiment 001; do not compare scores as a controlled improvement.",
              "- Validation both selects checkpoints and compares budgets. These single-user data cannot establish generalization to unseen people.",
              "- False motion counts are raw clip classifications, not live robot events/minute; STOP recall is not stopping latency.",
              "- No withheld training labels, test arrays, or test predictions enter training or checkpoint selection.",
              "- Every model, subset, source/configuration hash, history, and validation prediction is retained. No app model is activated.", ""]
    return "\n".join(lines)


def plot_summary(summary, output, config):
    os.environ.setdefault("MPLCONFIGDIR", str(output / "plot_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    entries = list(summary["conditions"].values())
    x = [e["training_clips"] for e in entries]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, field, title in zip(axes, ("accuracy", "no_command_to_motion_rate"),
                                  ("Validation accuracy", "NO_COMMAND classified as movement")):
        mean = [e[field]["mean"] for e in entries]
        sd = [e[field]["std_across_subset_and_model_seeds"] or 0 for e in entries]
        axis.errorbar(x, mean, yerr=sd, marker="o", capsize=4)
        axis.set(xlabel="Human labels used for training", title=title, ylim=(-0.04, 1.04), xticks=x)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=0.25)
    figure.suptitle(f"Supervised label-budget pilot: {config['optimizer_steps']} updates per run")
    figure.text(0.5, 0.02, "Same development validation session(s); bars = subset/model run SD. Not a test or robot result.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    figure.savefig(output / "label_budget_curve.png", dpi=180)
    figure.savefig(output / "label_budget_curve.pdf")
    plt.close(figure)


def run(config_path, output_root):
    config = json.loads(config_path.read_text())
    validate_config(config)
    folder = Path(config["data"])
    if not folder.is_absolute():
        folder = PROJECT / folder
    dataset, manifest = development_snapshot(folder)
    pool = original_pool(dataset, manifest)
    plans = {seed: nested_subsets(pool, manifest, config, seed) for seed in config["seeds"]}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"state": "preparing", "completed_trials": 0})
    results = []
    try:
        write_json(output / "config.json", config)
        write_json(output / "manifest.json", manifest)
        records = {r["file"]: r for r in manifest["records"]}
        selections = [{"seed": seed, "condition": name, "training_clips": len(samples),
                       "labels_by_class": dict(Counter(LABELS[s[1]] for s in samples)),
                       "sessions": dict(Counter(records[s[2]]["session"] for s in samples)),
                       "files": [s[2] for s in samples]}
                      for seed, conditions in plans.items() for name, samples in conditions.items()]
        write_json(output / "selections.json", selections)
        tx, ty = tensors(dataset["train"])
        vx, vy = tensors(dataset["validation"])
        np.savez_compressed(output / "development_features.npz", train_x=tx.numpy(), train_y=ty.numpy(),
                            validation_x=vx.numpy(), validation_y=vy.numpy())
        hashes = {}
        sources = (*SOURCE_FILES, "research/experiments/label_budget.py",
                   "research/experiments/test_label_budget.py", "research/LESSON_02.md")
        for name in sources:
            destination = output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, destination)
            hashes[name] = sha256(destination)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True)
        status = subprocess.run(["git", "status", "--short"], cwd=PROJECT, capture_output=True, text=True)
        environment = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                       "matplotlib": importlib.metadata.version("matplotlib"),
                       "platform": platform.platform(), "machine": platform.machine(), "device": "cpu",
                       "threads": config["threads"], "git_commit": commit.stdout.strip(),
                       "git_status_at_start": status.stdout, "source_sha256": hashes,
                       "artifact_sha256": {name: sha256(output / name) for name in
                                           ("config.json", "manifest.json", "selections.json", "development_features.npz")}}
        write_json(output / "environment.json", environment)
        print(f"Study folder: {output}", flush=True)
        print(f"Original pool: {len(pool)}; validation: {len(dataset['validation'])}; corrections excluded: "
              f"{len(dataset['train']) - len(pool)}", flush=True)
        for seed, conditions in plans.items():
            for name, selected in conditions.items():
                trial = output / f"{name}_seed_{seed}"
                trial.mkdir()
                model, metrics, history, probabilities = train_steps(selected, dataset["validation"], config, seed)
                metrics["condition"] = name
                write_json(trial / "metrics.json", metrics)
                write_json(trial / "history.json", history)
                write_json(trial / "session_metrics.json", session_metrics(probabilities, dataset["validation"], manifest))
                write_json(trial / "validation_predictions.json", [
                    {"file": sample[2], "actual": LABELS[sample[1]],
                     "predicted": LABELS[int(probability.argmax())], "scores": probability.tolist()}
                    for sample, probability in zip(dataset["validation"], probabilities)])
                development_files = [s[2] for s in selected] + [s[2] for s in dataset["validation"]]
                torch.save({"state_dict": model.state_dict(), "schema_version": VERSION, "convention": CONVENTION,
                            "labels": list(LABELS), "frames": FRAMES, "seconds": SECONDS, "seed": seed,
                            "best_step": metrics["best_step"], "study": config["experiment_id"], "condition": name,
                            "development_files": development_files,
                            "development_sessions": sorted({records[f]["session"] for f in development_files}),
                            "training_label_files": [s[2] for s in selected],
                            "manifest_sha256": environment["artifact_sha256"]["manifest.json"]}, trial / "model.pt")
                results.append(metrics)
                write_json(output / "trials.json", results)
                write_json(output / "status.json", {"state": "running", "completed_trials": len(results)})
                print(f"{name:20s} seed={seed}: accuracy={metrics['accuracy']:.2%}, F1={metrics['macro_f1']:.4f}, "
                      f"best update={metrics['best_step']}, train={metrics['training_seconds']:.2f}s", flush=True)
        summary = summarize(results)
        summary["trials"] = results
        write_json(output / "summary.json", summary)
        (output / "report.md").write_text(markdown_report(summary, manifest, pool, config))
        plot_summary(summary, output, config)
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
