"""Evaluate all saved v2 candidates on new validation sessions, without fitting.

Leaves commands_v2.py and its archived training source unchanged. No test
payload, optimizer, candidate selection, gate tuning or camera activation.
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

from .commands_v2 import (DATA, LABELS, SOURCE_FILES, contract, development_snapshot,
                          load_model, metrics, predictions, summarize as model_summary)
from .run import PROJECT, sha256, write_json

DEFAULT_STUDY = PROJECT / "research/experiments/runs/commands_v2/20261008T025353704748Z"


def new_cohort(dataset, current, previous):
    """Require intact prior inputs, wholly new original sessions and unique content."""
    if current["contract"] != previous["contract"]:
        raise ValueError("New data and frozen study have different command contracts.")
    old = {r["file"]: r for r in previous["records"]}
    now = {r["file"]: r for r in current["records"]}
    for name, record in old.items():
        if name not in now or now[name]["sha256"] != record["sha256"]:
            raise ValueError(f"Frozen development recording missing or changed: {name}")
    records = [r for r in current["records"] if r["file"] not in old]
    if not records:
        raise ValueError("No new validation recordings relative to the frozen study.")
    old_sessions = {r["session"] for r in old.values()}
    old_contents = {r["raw_content_sha256"] for r in old.values()}
    seen = set()
    for record in records:
        if record["split"] != "validation" or record["source"] != "recording":
            raise ValueError("New cohort must contain original validation recordings only.")
        if record["session"] in old_sessions:
            raise ValueError("New recording belongs to a frozen development session.")
        identity = record["raw_content_sha256"]
        if identity in old_contents or identity in seen:
            raise ValueError("New cohort duplicates raw development recording contents.")
        seen.add(identity)
    for session in {r["session"] for r in records}:
        counts = Counter(r["label"] for r in records if r["session"] == session)
        if any(counts[label] < 5 for label in LABELS):
            raise ValueError("Each new validation session needs at least five of every label.")
    names = {r["file"] for r in records}
    samples = [sample for sample in dataset["validation"] if sample[2] in names]
    if len(samples) != len(records) or {s[2] for s in samples} != names:
        raise ValueError("New feature rows and recording manifest differ.")
    manifest = {"contract": contract(), "records": records,
                "summary": {"clips": len(records), "sessions": sorted({r["session"] for r in records}),
                            "labels": dict(Counter(r["label"] for r in records)),
                            "tracking_quality_failures": [r["file"] for r in records if not r["passes_tracking_quality"]]},
                "scope": "new validation sessions only; no test payloads"}
    return samples, manifest


def check_study(study):
    env = json.loads((study / "environment.json").read_text())
    for name, digest in env["input_sha256"].items():
        if sha256(study / name) != digest:
            raise ValueError(f"Frozen study input changed: {name}")
    for name, digest in env["source_sha256"].items():
        if sha256(study / "source" / name) != digest:
            raise ValueError(f"Archived training source changed: {name}")
        if name.endswith(".py") and sha256(PROJECT / name) != digest:
            raise ValueError(f"Current features/model/evaluation source changed: {name}")
    if json.loads((study / "contract.json").read_text()) != contract():
        raise ValueError("Study uses a different command contract.")
    config = json.loads((study / "config.json").read_text())
    seeds = config["seeds"]
    old_trials = json.loads((study / "trials.json").read_text())
    if (not seeds or len(set(seeds)) != len(seeds) or len(old_trials) != len(seeds)
            or {r["seed"] for r in old_trials} != set(seeds)
            or json.loads((study / "status.json").read_text()) != {"state": "complete", "completed_trials": len(seeds)}):
        raise ValueError("Training study lacks a complete prespecified candidate list.")
    verified = json.loads((study / "verification.json").read_text())["checkpoints_sha256"]
    expected = {f"seed_{seed}/model.pt" for seed in seeds}
    if set(verified) != expected:
        raise ValueError("Verified checkpoint list differs from the training configuration.")
    candidates = []
    for seed in seeds:
        name = f"seed_{seed}/model.pt"
        if sha256(study / name) != verified[name]:
            raise ValueError("Saved checkpoint differs from its post-training verification hash.")
        candidates.append({"seed": seed, "file": name, "sha256": verified[name]})
    return env, candidates, old_trials


def aggregate(trials, old_trials):
    result = model_summary(trials)
    old = {r["seed"]: r for r in old_trials}
    result["old_selection_session_metrics"] = model_summary(old_trials)["metrics"]
    result["paired_changes_from_selection_session"] = {
        field: {"by_seed": {str(r["seed"]): r[field]-old[r["seed"]][field] for r in trials},
                "mean": float(np.mean([r[field]-old[r["seed"]][field] for r in trials]))}
        for field in ("accuracy", "macro_f1", "no_command_to_motion_rate", "stop_recall")}
    result["class_recall_comparison"] = {
        label: {"new_support_per_model": trials[0]["per_class"][label]["support"],
                "old_support_per_model": old_trials[0]["per_class"][label]["support"],
                "new_mean": float(np.mean([r["per_class"][label]["recall"] for r in trials])),
                "old_mean": float(np.mean([r["per_class"][label]["recall"] for r in old_trials]))}
        for label in LABELS}
    result["interpretation"] = (
        "Frozen-weight, single-user new-session development validation. No retraining or reselection. "
        "Old and new sessions are separate; descriptive changes are not causal improvements. "
        "Run SD is training variability on shared clips, not a population interval. "
        "No final test, gate/robot outcome, unseen-user reliability, SSL or annotation savings.")
    return result


def markdown_report(summary, manifest, study):
    lines = ["# Eight-command models: new-session validation", "",
             f"Original study: `{study.name}`. All declared candidates; no retraining/reselection.",
             f"New cohort: {manifest['summary']['clips']} clips, sessions {manifest['summary']['sessions']}.",
             f"Class counts: {manifest['summary']['labels']}.", "",
             "| Seed | Old selection-session accuracy | New-session accuracy | New macro F1 | NO_COMMAND → motion | STOP → motion |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    old_accuracy = {r["seed"]: r["accuracy"] for r in summary["trials"]}
    # Recover old values from explicitly saved paired differences, not a pooled set.
    deltas = summary["paired_changes_from_selection_session"]["accuracy"]["by_seed"]
    for r in summary["trials"]:
        old_accuracy[r["seed"]] -= deltas[str(r["seed"])]
        lines.append(f"| {r['seed']} | {100*old_accuracy[r['seed']]:.2f}% | {100*r['accuracy']:.2f}% | "
                     f"{r['macro_f1']:.4f} | {r['no_command_to_motion_count']} | {r['stop_to_motion_count']} |")
    for field in ("accuracy", "macro_f1", "no_command_to_motion_rate", "stop_recall"):
        values = summary["metrics"][field]
        sd = "n/a" if values["run_sd"] is None else f"{values['run_sd']:.4f}"
        lines.append(f"\n{field}: mean {values['mean']:.4f}, run SD {sd}.")
    lines += ["", "| Class | Old recall mean | New recall mean | New clips per model |",
              "| --- | ---: | ---: | ---: |"]
    for label, entry in summary["class_recall_comparison"].items():
        lines.append(f"| {label} | {100*entry['old_mean']:.2f}% | {100*entry['new_mean']:.2f}% | {entry['new_support_per_model']} |")
    lines += ["", summary["interpretation"], "",
              "Each prediction is raw classification, not an issued robot action. "
              "Each new session has its own metrics; no pooling with checkpoint-selection validation. "
              "These inspected sessions are development data, not a reusable untouched final benchmark.", ""]
    return "\n".join(lines)


def run(study=DEFAULT_STUDY, folder=DATA):
    study = Path(study).resolve()
    env, candidates, old_trials = check_study(study)
    previous = json.loads((study / "manifest.json").read_text())
    dataset, current = development_snapshot(folder)
    samples, manifest = new_cohort(dataset, current, previous)
    x = torch.from_numpy(np.stack([s[0] for s in samples]))
    targets = np.asarray([s[1] for s in samples])
    with np.load(study / "development_features.npz", allow_pickle=False) as a:
        old_x = torch.from_numpy(a["validation_x"].copy())
        old_y = a["validation_y"].copy()
    old_val = [r for r in previous["records"] if r["split"] == "validation"]
    old_sessions = sorted({r["session"] for r in previous["records"]})
    new_sessions = manifest["summary"]["sessions"]
    output = study / "session_evaluations" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"state": "preparing", "completed_trials": 0})
    trials = []
    try:
        write_json(output / "manifest.json", manifest)
        write_json(output / "evaluation_plan.json", {"study": str(study), "checkpoints": candidates,
                   "evaluation": "raw classification on new validation only", "retraining": False,
                   "checkpoint_selection": False, "primary_outcome": "macro F1 across every declared seed"})
        np.savez_compressed(output / "new_validation_features.npz", x=x.numpy(), y=targets)
        sources = (*SOURCE_FILES, "research/experiments/evaluate_v2_sessions.py",
                   "research/experiments/test_evaluate_v2_sessions.py", "research/experiments/run.py")
        source_hashes = {}
        for name in sources:
            destination = output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, destination)
            source_hashes[name] = sha256(destination)
        write_json(output / "environment.json", {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                   "platform": platform.platform(), "device": "cpu", "threads": 2, "source_sha256": source_hashes,
                   "original_input_sha256": env["input_sha256"],
                   "artifact_sha256": {name: sha256(output / name) for name in
                                        ("manifest.json", "evaluation_plan.json", "new_validation_features.npz")}})
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        print(f"Evaluation folder: {output}", flush=True)
        for candidate in candidates:
            seed = candidate["seed"]
            model, checkpoint = load_model(study / candidate["file"])
            if (checkpoint.get("seed") != seed or checkpoint.get("manifest_sha256") != sha256(study / "manifest.json")
                    or checkpoint.get("development_files") != [r["file"] for r in previous["records"]]
                    or checkpoint.get("development_sessions") != old_sessions
                    or set(new_sessions).intersection(checkpoint["development_sessions"])):
                raise ValueError("Checkpoint provenance differs from the frozen study or overlaps new sessions.")
            with torch.inference_mode():
                old_p = model(old_x).softmax(1).numpy()
                p = model(x).softmax(1).numpy()
            saved = json.loads((study / f"seed_{seed}/validation_predictions.json").read_text())
            if [r["file"] for r in saved] != [r["file"] for r in old_val]:
                raise ValueError("Original prediction rows differ from the training manifest.")
            np.testing.assert_allclose(old_p, [r["scores"] for r in saved], rtol=1e-6, atol=1e-7)
            old_metrics = metrics(old_p, old_y)
            recorded_metrics = next(r for r in old_trials if r["seed"] == seed)
            if old_metrics != {key: recorded_metrics[key] for key in old_metrics}:
                raise ValueError("Checkpoint does not reproduce original metrics.")
            result = metrics(p, targets)
            result.update(seed=seed, checkpoint_sha256=candidate["sha256"], sessions={})
            for session in new_sessions:
                names = {r["file"] for r in manifest["records"] if r["session"] == session}
                indices = [i for i, sample in enumerate(samples) if sample[2] in names]
                result["sessions"][session] = metrics(p[indices], targets[indices])
            trial = output / f"seed_{seed}"
            trial.mkdir()
            write_json(trial / "metrics.json", result)
            write_json(trial / "predictions.json", predictions(samples, p))
            trials.append(result)
            write_json(output / "trials.json", trials)
            write_json(output / "status.json", {"state": "running", "completed_trials": len(trials)})
            print(f"seed={seed}: new-session accuracy={result['accuracy']:.2%}, F1={result['macro_f1']:.4f}", flush=True)
        for record in (*previous["records"], *manifest["records"]):
            if sha256(record["file"]) != record["sha256"]:
                raise ValueError("A development recording changed during evaluation.")
        for candidate in candidates:
            if sha256(study / candidate["file"]) != candidate["sha256"]:
                raise ValueError("A frozen checkpoint changed during evaluation.")
        summary = aggregate(trials, old_trials)
        summary["per_session"] = {session: aggregate([{**r, **r["sessions"][session]} for r in trials], old_trials)
                                  for session in new_sessions}
        write_json(output / "summary.json", summary)
        (output / "report.md").write_text(markdown_report(summary, manifest, study))
        write_json(output / "status.json", {"state": "complete", "completed_trials": len(trials)})
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "completed_trials": len(trials), "error": str(error)})
        raise
    print(f"Read results: {output / 'report.md'}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    try:
        run(args.study, args.data)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
