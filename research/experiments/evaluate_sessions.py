"""Evaluate experiment 002's frozen checkpoints on wholly new validation sessions.

No optimizer, checkpoint selection, test prediction, or camera activation.
Each invocation preserves a separate evaluation beneath the original study.
"""

import argparse
import json
import platform
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from gesture_commands.data import LABELS
from gesture_commands.model import load_model
from .label_budget import session_metrics, validate_config
from .run import PROJECT, development_snapshot, sha256, validation_metrics, write_json

DEFAULT_STUDY = PROJECT / "research/experiments/runs/20261008T005037258990Z"
FIELDS = ("accuracy", "macro_f1", "nll", "no_command_to_motion_rate",
          "stop_recall", "stop_to_motion_count")


def new_validation_cohort(dataset, current, previous):
    """File identity alone is insufficient: check session and raw-content overlap."""
    old = {r["file"]: r for r in previous["records"]}
    now = {r["file"]: r for r in current["records"]}
    for name, record in old.items():
        if name not in now or now[name]["sha256"] != record["sha256"]:
            raise ValueError(f"Frozen development recording missing or changed: {name}")
    records = [r for r in current["records"] if r["file"] not in old]
    if not records:
        raise ValueError("No new validation recordings relative to this study.")
    sessions = {r["session"] for r in old.values()}
    contents = {r["raw_content_sha256"] for r in old.values()}
    seen = set()
    for r in records:
        if r["split"] != "validation" or r["source"] != "recording":
            raise ValueError("New development files must be original validation recordings.")
        if r["session"] in sessions:
            raise ValueError("New evaluation must use wholly new sessions.")
        identity = r["raw_content_sha256"]
        if identity in contents or identity in seen:
            raise ValueError("New evaluation contains duplicate raw recording contents.")
        seen.add(identity)
    names = {r["file"] for r in records}
    samples = [s for s in dataset["validation"] if s[2] in names]
    if len(samples) != len(records):
        raise ValueError("New cohort and feature rows differ.")
    # All six classes must be present in each session for comparable macro F1.
    for session in {r["session"] for r in records}:
        labels = {r["label"] for r in records if r["session"] == session}
        if labels != set(LABELS):
            raise ValueError(f"Session {session} does not include all six labels.")
    return samples, {**current, "records": records,
                     "summary": {"clips": len(records),
                                 "sessions": len({r["session"] for r in records}),
                                 "labels": dict(Counter(r["label"] for r in records)),
                                 "tracking_quality_failures": sum(not r["passes_tracking_quality"] for r in records)},
                     "scope": "new original validation sessions only; no test payloads"}


def summarize(results, old_trials):
    old = {(r["condition"], r["seed"]): r for r in old_trials}
    reference = {r["seed"]: r for r in results if r["condition"] == "all_original"}
    conditions = {}
    for name in dict.fromkeys(r["condition"] for r in results):
        group = [r for r in results if r["condition"] == name]
        entry = {"training_clips": group[0]["training_clips"]}
        for field in FIELDS:
            values = [r[field] for r in group]
            entry[field] = {"mean": float(np.mean(values)),
                            "run_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        entry["per_class_mean_recall"] = {
            label: float(np.mean([r["per_class"][label]["recall"] for r in group])) for label in LABELS}
        entry["old_session_accuracy_mean"] = float(np.mean([old[name, r["seed"]]["accuracy"] for r in group]))
        entry["accuracy_change_from_old_session_pp"] = 100 * (entry["accuracy"]["mean"] - entry["old_session_accuracy_mean"])
        entry["paired_difference_from_all_original"] = {
            field: {"mean": float(np.mean([r[field] - reference[r["seed"]][field] for r in group])),
                    "by_seed": {str(r["seed"]): r[field] - reference[r["seed"]][field] for r in group}}
            for field in ("accuracy", "macro_f1", "no_command_to_motion_rate")}
        conditions[name] = entry
    return {"conditions": conditions,
            "interpretation": "Frozen-model, single-user new-session development evaluation. "
            "Run SD is model/subset variability on shared clips, not a population interval. "
            "Old and new sessions remain separate. No retraining, test payload, gate evaluation, "
            "robot result, or measured annotation saving."}


def report(summary, manifest, study):
    lines = ["# Experiment 002: frozen models on new validation sessions", "",
             f"Original study: `{study.name}`. No retraining or checkpoint selection.",
             f"New cohort: {manifest['summary']['clips']} clips in {manifest['summary']['sessions']} sessions.",
             f"Labels: {manifest['summary']['labels']}.", "",
             "| Training labels | Old accuracy | New accuracy mean ± run SD | New macro F1 | NO_COMMAND → motion | STOP recall |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for entry in summary["conditions"].values():
        sd = entry["accuracy"]["run_sd"]
        spread = "n/a" if sd is None else f"{100 * sd:.2f} pp"
        lines.append(f"| {entry['training_clips']} | {100 * entry['old_session_accuracy_mean']:.2f}% | "
                     f"{100 * entry['accuracy']['mean']:.2f}% ± {spread} | {entry['macro_f1']['mean']:.4f} | "
                     f"{100 * entry['no_command_to_motion_rate']['mean']:.2f}% | {100 * entry['stop_recall']['mean']:.2f}% |")
    lines += ["", "## Per-class mean recall", "",
              "| Training labels | " + " | ".join(LABELS) + " |",
              "| ---: | " + " | ".join(["---:"] * len(LABELS)) + " |"]
    for entry in summary["conditions"].values():
        lines.append(f"| {entry['training_clips']} | " + " | ".join(
            f"{100 * entry['per_class_mean_recall'][label]:.2f}%" for label in LABELS) + " |")
    lines += ["", summary["interpretation"], "",
              "All seeds and session-specific confusion matrices are retained in trials.json. "
              "Raw false-motion counts are classification errors, not executed robot actions. "
              "A tracking-quality pass does not establish correct semantic labels. "
              "These inspected sessions must not be reused as an untouched final benchmark.", ""]
    return "\n".join(lines)


def run(study, folder):
    study = Path(study).resolve()
    config = json.loads((study / "config.json").read_text())
    validate_config(config)
    env = json.loads((study / "environment.json").read_text())
    for name, expected in env["artifact_sha256"].items():
        if sha256(study / name) != expected:
            raise ValueError(f"Frozen study artifact changed: {name}")
    for name in ("gesture_commands/data.py", "gesture_commands/model.py", "research/experiments/run.py"):
        if sha256(PROJECT / name) != env["source_sha256"][name]:
            raise ValueError(f"Frozen preprocessing/model/metrics source changed: {name}")
    previous = json.loads((study / "manifest.json").read_text())
    old_trials = json.loads((study / "trials.json").read_text())
    selections = json.loads((study / "selections.json").read_text())
    names = [f"labels_{n}_per_class" for n in config["labels_per_class"]]
    if config["include_all_original_reference"]:
        names.append("all_original")
    if "all_original" not in names:
        raise ValueError("New-session comparison requires the all-original reference.")
    expected = {(name, seed) for name in names for seed in config["seeds"]}
    for rows in (old_trials, selections):
        if len(rows) != len(expected) or {(r["condition"], r["seed"]) for r in rows} != expected:
            raise ValueError("Frozen trial list differs from the prespecified conditions/seeds.")
    status = json.loads((study / "status.json").read_text())
    if status != {"state": "complete", "completed_trials": len(expected)}:
        raise ValueError("Training study did not complete all prespecified trials.")
    dataset, current = development_snapshot(folder)
    samples, manifest = new_validation_cohort(dataset, current, previous)
    x = torch.from_numpy(np.stack([s[0] for s in samples]))
    targets = np.asarray([s[1] for s in samples])
    new_sessions = {r["session"] for r in manifest["records"]}
    old_records = {r["file"]: r for r in previous["records"]}
    old_val_files = [r["file"] for r in previous["records"] if r["split"] == "validation"]
    with np.load(study / "development_features.npz", allow_pickle=False) as a:
        old_x = torch.from_numpy(a["validation_x"].copy())
    trial_records = []
    for selection in selections:
        name, seed = selection["condition"], selection["seed"]
        path = study / f"{name}_seed_{seed}" / "model.pt"
        trial_records.append({"condition": name, "seed": seed, "file": str(path.relative_to(PROJECT)),
                              "sha256": sha256(path), "training_clips": selection["training_clips"]})
    output = study / "session_evaluations" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    write_json(output / "status.json", {"state": "preparing", "completed_trials": 0})
    try:
        write_json(output / "manifest.json", manifest)
        write_json(output / "evaluation_plan.json", {
            "study": str(study.relative_to(PROJECT)), "checkpoints": trial_records,
            "conditions": names, "seeds": config["seeds"], "retraining": False,
            "checkpoint_selection": False, "evaluation": "raw classification; new validation only",
            "primary_comparison": "macro F1 paired difference from all_original"})
        np.savez_compressed(output / "new_validation_features.npz", x=x.numpy(), y=targets)
        source_hashes = {}
        for source in ("research/experiments/evaluate_sessions.py", "research/experiments/test_evaluate_sessions.py",
                       "research/PROTOCOL.md", "gesture_commands/data.py", "gesture_commands/model.py",
                       "research/experiments/run.py", "research/experiments/label_budget.py"):
            destination = output / "source" / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / source, destination)
            source_hashes[source] = sha256(destination)
        write_json(output / "environment.json", {
            "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "device": "cpu", "threads": 2,
            "source_sha256": source_hashes,
            "original_study_artifact_sha256": env["artifact_sha256"],
            "artifact_sha256": {f: sha256(output / f) for f in
                                ("manifest.json", "evaluation_plan.json", "new_validation_features.npz")}})
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        print(f"Evaluation folder: {output}", flush=True)
        for selection, record in zip(selections, trial_records):
            name, seed = selection["condition"], selection["seed"]
            path = PROJECT / record["file"]
            model, checkpoint = load_model(path)
            files = selection["files"] + old_val_files
            sessions = sorted({old_records[f]["session"] for f in files})
            if (checkpoint.get("study") != config["experiment_id"] or checkpoint.get("condition") != name
                    or checkpoint.get("seed") != seed or checkpoint.get("training_label_files") != selection["files"]
                    or checkpoint.get("development_files") != files
                    or checkpoint.get("development_sessions") != sessions
                    or checkpoint.get("manifest_sha256") != sha256(study / "manifest.json")
                    or new_sessions.intersection(sessions)):
                raise ValueError("Checkpoint provenance differs from frozen study or overlaps new sessions.")
            with torch.inference_mode():
                old_probabilities = model(old_x).softmax(1).numpy()
                probabilities = model(x).softmax(1).numpy()
            old_predictions = json.loads((path.parent / "validation_predictions.json").read_text())
            if [r["file"] for r in old_predictions] != old_val_files:
                raise ValueError("Original prediction rows differ from frozen manifest.")
            np.testing.assert_allclose(old_probabilities, [r["scores"] for r in old_predictions],
                                       rtol=1e-5, atol=1e-7,
                                       err_msg="Checkpoint does not reproduce its original frozen predictions.")
            if sha256(path) != record["sha256"]:
                raise ValueError("Checkpoint changed during evaluation.")
            metrics = validation_metrics(probabilities, targets)
            metrics.update(condition=name, seed=seed, training_clips=record["training_clips"],
                           checkpoint_sha256=record["sha256"],
                           sessions=session_metrics(probabilities, samples, manifest))
            trial = output / f"{name}_seed_{seed}"
            trial.mkdir()
            write_json(trial / "metrics.json", metrics)
            write_json(trial / "predictions.json", [
                {"file": sample[2], "actual": LABELS[sample[1]],
                 "predicted": LABELS[int(p.argmax())], "scores": p.tolist()}
                for sample, p in zip(samples, probabilities)])
            results.append(metrics)
            write_json(output / "trials.json", results)
            write_json(output / "status.json", {"state": "running", "completed_trials": len(results)})
            print(f"{name:20s} seed={seed}: accuracy={metrics['accuracy']:.2%}, F1={metrics['macro_f1']:.4f}", flush=True)
        # Confirm the entire frozen evaluation, not only one read, left inputs intact.
        for record in (*previous["records"], *manifest["records"], *trial_records):
            if sha256(PROJECT / record["file"]) != record["sha256"]:
                raise ValueError(f"Input/checkpoint changed: {record['file']}")
        summary = summarize(results, old_trials)
        summary["per_session"] = {
            session: summarize([{**r, **r["sessions"][session]} for r in results], old_trials)
            for session in sorted(new_sessions)}
        write_json(output / "summary.json", summary)
        (output / "report.md").write_text(report(summary, manifest, study))
        write_json(output / "status.json", {"state": "complete", "completed_trials": len(results)})
    except BaseException as error:
        write_json(output / "status.json", {"state": "interrupted_or_failed", "completed_trials": len(results),
                                           "error": f"{type(error).__name__}: {error}"})
        raise
    print(f"Read results: {output / 'report.md'}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--data", type=Path, default=PROJECT / "gesture_commands/data")
    args = parser.parse_args()
    run(args.study, args.data)


if __name__ == "__main__":
    main()
