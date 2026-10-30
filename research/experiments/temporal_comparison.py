"""Experiment 008: fixed-budget CNN, set, LSTM and BiLSTM development comparison.

Research-only checkpoints. Reuses pinned v3 features and the original CNN fitter.
Run: .venv/bin/python -m research.experiments.temporal_comparison train
"""

import argparse
import copy
import hashlib
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import boundary_train as fitting
from . import commands_v2 as v2
from . import commands_v3_train as training
from .run import PROJECT, sha256, write_json

STUDY = training.OUTPUT / "20261012T205727638067Z"
OUTPUT = PROJECT / "research/experiments/runs/temporal_comparison"
CONDITIONS = ("cnn", "pose_set", "feature_set", "lstm", "bilstm")
PARAMETERS = dict(zip(CONDITIONS, (48520, 48792, 49176, 48944, 49256)))
SOURCES = tuple(dict.fromkeys((*training.SOURCES,
    "research/experiments/temporal_comparison.py",
    "research/experiments/test_temporal_comparison.py",
    "research/TEMPORAL_COMPARISON_PROTOCOL.md")))


class FrameSet(nn.Module):
    """A collection of poses/features with mean pooling, without frame ordering."""

    def __init__(self, pose_only=False):
        super().__init__()
        self.pose_only = pose_only
        self.frame_encoder = nn.Sequential(
            nn.Linear(64 if pose_only else 67, 128), nn.ReLU(),
            nn.Linear(128, 176), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(176, 96), nn.ReLU(),
                                  nn.Dropout(.2), nn.Linear(96, 8))

    def forward(self, x):
        if self.pose_only:
            x = torch.cat((x[:, :, :63], x[:, :, 66:67]), dim=2)
        return self.head(self.frame_encoder(x).mean(dim=1))


class RecurrentGesture(nn.Module):
    """Full supplied window; bidirectional processing uses captured samples only."""

    def __init__(self, bidirectional=False):
        super().__init__()
        hidden = 36 if bidirectional else 62
        width = hidden * (2 if bidirectional else 1)
        self.recurrent = nn.LSTM(67, hidden, batch_first=True,
                                 bidirectional=bidirectional)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(4), nn.Flatten(),
            nn.Linear(width * 4, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, 8))

    def forward(self, x):
        sequence, _ = self.recurrent(x)
        return self.head(sequence.transpose(1, 2))


def build_model(condition):
    if condition == "cnn":
        model = v2.build_model()
    elif condition in ("pose_set", "feature_set"):
        model = FrameSet(pose_only=condition == "pose_set")
    elif condition in ("lstm", "bilstm"):
        model = RecurrentGesture(bidirectional=condition == "bilstm")
    else:
        raise ValueError("Unknown declared architecture.")
    if sum(p.numel() for p in model.parameters()) != PARAMETERS[condition]:
        raise ValueError("Architecture differs from the declared parameter count.")
    return model


def fit(condition, arrays, seed, config):
    """Same recipe as boundary_train; CNN calls that existing fitter directly."""
    if condition == "cnn":
        return fitting.fit_candidate(arrays["train_x"], arrays["train_y"],
            arrays["validation_x"], arrays["validation_y"], seed, config)
    tx, ty = torch.from_numpy(arrays["train_x"]), torch.from_numpy(arrays["train_y"])
    vx, vy = torch.from_numpy(arrays["validation_x"]), torch.from_numpy(arrays["validation_y"])
    counts = np.bincount(arrays["train_y"], minlength=len(v2.LABELS))
    if np.any(counts < 5):
        raise ValueError("Need at least five training clips per class.")
    torch.manual_seed(seed)
    model = build_model(condition)
    initial_sha = fitting.state_hash(model)
    rng = torch.Generator().manual_seed(seed + 1_000_000)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    weights = torch.tensor(counts.sum() / (len(v2.LABELS) * counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    best_loss, best_state, best_step = float("inf"), None, None
    history, batch_hash = [], hashlib.sha256()
    started = time.perf_counter()
    for step in range(1, config["optimizer_steps"] + 1):
        model.train()
        indices = torch.randint(len(tx), (config["batch_size"],), generator=rng)
        batch_hash.update(indices.numpy().tobytes())
        optimizer.zero_grad()
        loss = loss_fn(model(tx[indices]), ty[indices])
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        if step % config["validation_every_steps"] == 0 or step == config["optimizer_steps"]:
            model.eval()
            with torch.inference_mode():
                val_loss = nn.functional.cross_entropy(model(vx), vy).item()
            if not np.isfinite(val_loss):
                raise ValueError("Non-finite checkpoint-selection loss.")
            history.append({"step": step, "selection_loss": val_loss})
            if val_loss < best_loss:
                best_loss, best_state, best_step = val_loss, copy.deepcopy(model.state_dict()), step
    model.load_state_dict(best_state)
    model.eval()
    return model, history, {
        "seed": seed, "best_step": best_step, "training_seconds": time.perf_counter() - started,
        "optimizer_steps": config["optimizer_steps"], "validation_checks": len(history),
        "training_clips": len(tx), "training_class_counts": counts.tolist(),
        "initial_state_sha256": initial_sha, "batch_indices_sha256": batch_hash.hexdigest()}


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"experiment_id": "008_temporal_comparison", "schema_version": 3,
        "feature_version": 1, "taxonomy": training.collection.TAXONOMY,
        "labels": list(v2.LABELS), "convention": v2.CONVENTION,
        "frames": v2.FRAMES, "features": v2.FEATURES, "seconds": v2.SECONDS}
    contract = checkpoint.get("command_contract", {})
    try:
        expected["command_contract"] = training.collection.contract(contract["up_style"])
    except (KeyError, ValueError) as error:
        raise ValueError("Checkpoint lacks the v3 contract.") from error
    if any(checkpoint.get(k) != value for k, value in expected.items()):
        raise ValueError("Incompatible research-only temporal checkpoint.")
    model = build_model(checkpoint.get("condition"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def window_latency(model, x, warmup, repeats):
    elapsed = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        for _ in range(repeats):
            started = time.perf_counter_ns()
            model(x)
            elapsed.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_ms": float(np.median(elapsed)), "p90_ms": float(np.percentile(elapsed, 90)),
        "warmup": warmup, "timed_passes": repeats,
        "scope": "CPU forward pass for one preprocessed full window; acquisition/tracker excluded."}


def worker(folder):
    folder = Path(folder)
    output = folder.parent
    config = fitting.read_json(output / "config.json")
    manifest = fitting.read_json(output / "manifest.json")
    job = fitting.read_json(folder / "job.json")
    condition, seed = job["condition"], job["seed"]
    if condition not in config["conditions"] or seed not in config["training"]["seeds"]:
        raise ValueError("Worker outside the declared comparison.")
    env = fitting.read_json(output / "environment.json")
    if any(sha256(output / name) != digest for name, digest in env["input_sha256"].items()):
        raise ValueError("Worker inputs changed.")
    if any(sha256(PROJECT / name) != digest for name, digest in env["source_sha256"].items()
           if name.endswith(".py")):
        raise ValueError("Executed Python sources differ from the archived study.")
    torch.set_num_threads(config["training"]["threads"])
    with np.load(output / "development_features.npz", allow_pickle=False) as a:
        arrays = {name: a[name].copy() for name in a.files}
    model, history, stats = fit(condition, arrays, seed, config["training"])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats["process_peak_rss_bytes_through_fit"] = int(rss if sys.platform == "darwin" else rss * 1024)
    stats["peak_rss_scope"] = "Fresh worker process through fit, including interpreter/Torch/data/optimizer."
    torch.save({"state_dict": model.state_dict(), "experiment_id": config["experiment_id"],
        "schema_version": 3, "feature_version": 1, "taxonomy": training.collection.TAXONOMY,
        "labels": list(v2.LABELS), "convention": v2.CONVENTION, "frames": v2.FRAMES,
        "features": v2.FEATURES, "seconds": v2.SECONDS, "condition": condition,
        "command_contract": manifest["contract"], "seed": seed, "best_step": stats["best_step"],
        "manifest_sha256": sha256(output / "manifest.json"),
        "config_sha256": sha256(output / "config.json"),
        "training_files": manifest["rows"]["train"],
        "validation_files": manifest["rows"]["validation"]}, folder / "model.pt")
    write_json(folder / "history.json", history)
    model, _ = load_model(folder / "model.pt")
    with torch.inference_mode():
        p = model(torch.from_numpy(arrays["validation_x"])).softmax(1).numpy()
    samples = [(x, int(y), f) for x, y, f in zip(arrays["validation_x"], arrays["validation_y"],
                                               manifest["rows"]["validation"])]
    trial = {**stats, "condition": condition, "parameters": PARAMETERS[condition],
        "checkpoint_sha256": sha256(folder / "model.pt"),
        "state_sha256": fitting.state_hash(model),
        "validation": fitting.metric_set(p, arrays["validation_y"]),
        "latency": window_latency(model, torch.from_numpy(arrays["validation_x"][:1]),
                                   config["latency_warmup"], config["latency_repeats"])}
    write_json(folder / "validation_predictions.json", v2.predictions(samples, p))
    write_json(folder / "metrics.json", trial)


def summarize(trials, config):
    by = {(r["condition"], r["seed"]): r for r in trials}
    seeds = config["training"]["seeds"]
    if len(by) != len(trials) or set(by) != {(c, s) for c in config["conditions"] for s in seeds}:
        raise ValueError("Require every declared condition and seed once.")
    result = {"conditions": {}, "paired_macro_f1_minus_cnn": {}}
    for condition in config["conditions"]:
        rows = [by[condition, seed] for seed in seeds]
        metrics = {}
        for field in fitting.FIELDS:
            values = [r["validation"][field] for r in rows]
            metrics[field] = {"mean": float(np.mean(values)),
                "seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        result["conditions"][condition] = {"parameters": PARAMETERS[condition], "metrics": metrics,
            "per_class_recall": {label: float(np.mean([r["validation"]["per_class"][label]["recall"]
                                   for r in rows])) for label in v2.LABELS},
            "training_seconds_mean": float(np.mean([r["training_seconds"] for r in rows])),
            "worker_wall_seconds_mean": float(np.mean([r["worker_wall_seconds"] for r in rows])),
            "window_latency_median_ms_mean": float(np.mean([r["latency"]["median_ms"] for r in rows])),
            "process_peak_rss_bytes_mean": float(np.mean([r["process_peak_rss_bytes_through_fit"] for r in rows]))}
        if condition != "cnn":
            differences = [by[condition, s]["validation"]["macro_f1"] - by["cnn", s]["validation"]["macro_f1"] for s in seeds]
            result["paired_macro_f1_minus_cnn"][condition] = {
                "seeds": seeds, "differences": differences, "mean": float(np.mean(differences)),
                "seed_sd": float(np.std(differences, ddof=1)) if len(seeds) > 1 else None}
    result["interpretation"] = "Checkpoint-selected, inspected single-user validation; fixed recipe, no test/label-saving/novelty/robot-reliability claim. Seed SD is not population uncertainty."
    return result


def report(summary):
    lines = ["# Experiment 008 — temporal architecture development comparison", "",
        "All five seeds retained. The same inspected validation selects checkpoints; this is not a final test.", "",
        "| Model | Parameters | Mean accuracy | Macro F1 | STOP recall | NONE→motion | Fit seconds/model | CPU window ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for condition, row in summary["conditions"].items():
        m = row["metrics"]
        lines.append(f"| {condition} | {row['parameters']:,} | {m['accuracy']['mean']:.2%} | {m['macro_f1']['mean']:.4f} | {m['stop_recall']['mean']:.2%} | {m['no_command_to_motion_rate']['mean']:.2%} | {row['training_seconds_mean']:.2f} | {row['window_latency_median_ms_mean']:.3f} |")
    lines += ["", "| Model | Paired mean macro F1 minus CNN | Five seed differences |", "| --- | ---: | --- |"]
    for condition, row in summary["paired_macro_f1_minus_cnn"].items():
        lines.append(f"| {condition} | {row['mean']:+.4f} | {', '.join(f'{d:+.4f}' for d in row['differences'])} |")
    lines += ["", "Window timings exclude tracking and acquisition. Peak RSS is total worker process memory, not isolated model memory.",
        "Equal updates/near-equal parameters do not equal equal compute. Full per-seed/class metrics and histories are preserved.",
        "No architecture-specific tuning, final-test access, camera activation or new labeling occurred."]
    return "\n".join(lines) + "\n"


def verify(output):
    output = Path(output)
    config = fitting.read_json(output / "config.json")
    env = fitting.read_json(output / "environment.json")
    manifest = fitting.read_json(output / "manifest.json")
    trials = fitting.read_json(output / "trials.json")
    for name, digest in env["input_sha256"].items():
        if sha256(output / name) != digest:
            raise ValueError("Archived comparison input changed.")
    for name, digest in env["source_sha256"].items():
        if sha256(output / "source" / name) != digest:
            raise ValueError("Archived comparison source changed.")
    for row in manifest["records"]:
        if sha256(row["file"]) != row["sha256"]:
            raise ValueError("Original recording changed.")
    torch.set_num_threads(config["training"]["threads"])
    with np.load(output / "development_features.npz", allow_pickle=False) as a:
        vx, vy = a["validation_x"].copy(), a["validation_y"].copy()
    counts = [manifest["counts"]["train"][label] for label in v2.LABELS]
    hashes, vectors, streams = {}, 0, {}
    for trial in trials:
        c, s = trial["condition"], trial["seed"]
        folder = output / f"{c}_seed_{s}"
        if sha256(folder / "model.pt") != trial["checkpoint_sha256"]:
            raise ValueError("Comparison checkpoint changed.")
        model, checkpoint = load_model(folder / "model.pt")
        expected = {"condition": c, "seed": s, "best_step": trial["best_step"],
            "command_contract": manifest["contract"], "manifest_sha256": sha256(output / "manifest.json"),
            "config_sha256": sha256(output / "config.json"),
            "training_files": manifest["rows"]["train"], "validation_files": manifest["rows"]["validation"]}
        if any(checkpoint.get(k) != v for k, v in expected.items()):
            raise ValueError("Checkpoint provenance changed.")
        rows = fitting.read_json(folder / "validation_predictions.json")
        if [r["file"] for r in rows] != manifest["rows"]["validation"] or [r["actual"] for r in rows] != [v2.LABELS[int(y)] for y in vy]:
            raise ValueError("Validation row order or human labels changed.")
        with torch.inference_mode():
            p = model(torch.from_numpy(vx)).softmax(1).numpy()
        np.testing.assert_allclose(p, [r["scores"] for r in rows], rtol=1e-6, atol=1e-7)
        if [r["predicted"] for r in rows] != [v2.LABELS[int(i)] for i in p.argmax(1)]:
            raise ValueError("Predicted labels changed.")
        if fitting.metric_set(p, vy) != trial["validation"] or fitting.read_json(folder / "metrics.json") != trial:
            raise ValueError("Saved metrics do not reproduce.")
        history = fitting.read_json(folder / "history.json")
        expected_steps = list(range(config["training"]["validation_every_steps"], config["training"]["optimizer_steps"] + 1,
                                    config["training"]["validation_every_steps"]))
        if not expected_steps or expected_steps[-1] != config["training"]["optimizer_steps"]:
            expected_steps.append(config["training"]["optimizer_steps"])
        chosen = min(history, key=lambda row: row["selection_loss"])
        if ([h["step"] for h in history] != expected_steps or chosen["step"] != trial["best_step"]
            or trial["validation_checks"] != len(history)
            or trial["optimizer_steps"] != config["training"]["optimizer_steps"]
            or trial["training_class_counts"] != counts
            or trial["training_clips"] != len(manifest["rows"]["train"])
            or trial["parameters"] != PARAMETERS[c]
            or trial["state_sha256"] != fitting.state_hash(model)
            or not np.isclose(chosen["selection_loss"], trial["validation"]["nll"], rtol=1e-6, atol=1e-7)):
            raise ValueError("Minimum-loss selection or training provenance changed.")
        streams.setdefault(s, set()).add(trial["batch_indices_sha256"])
        hashes[str((folder / "model.pt").relative_to(output))] = trial["checkpoint_sha256"]
        vectors += len(rows)
    if any(len(values) != 1 for values in streams.values()):
        raise ValueError("Batch streams are not paired across architectures.")
    if fitting.read_json(output / "summary.json") != summarize(trials, config):
        raise ValueError("Summary does not reproduce all declared trials.")
    source = Path(config["source_study"])
    old_checked = training.verify(source)
    for name, digest in config["original_checkpoint_sha256"].items():
        if sha256(source / name) != digest:
            raise ValueError("Prior checkpoint changed.")
    for trial in trials:
        if trial["condition"] != "cnn":
            continue
        old = fitting.read_json(source / f"seed_{trial['seed']}/metrics.json")
        for field in ("initial_state_sha256", "batch_indices_sha256", "best_step", "validation"):
            if trial[field] != old[field]:
                raise ValueError(f"CNN failed to reproduce prior {field}.")
        prior, _ = training.load_model(source / f"seed_{trial['seed']}/model.pt")
        if fitting.state_hash(prior) != trial["state_sha256"]:
            raise ValueError("CNN saved state differs from the reproduced prior fit.")
    return {"input_files_verified": len(manifest["records"]),
        "saved_prediction_vectors_verified": vectors, "checkpoints_sha256": hashes,
        "all_batch_streams_paired": True, "prior_cnn_states_reproduced": True,
        "original_checkpoint_verification": old_checked}


def run(study=STUDY, output_root=OUTPUT, isolated=True):
    study = Path(study)
    old_config = fitting.read_json(study / "config.json")
    torch.set_num_threads(old_config["threads"])
    previous = training.verify(study)
    _, manifest, _, arrays = training.snapshot()
    old_manifest = fitting.read_json(study / "manifest.json")
    if manifest != old_manifest:
        raise ValueError("Current v3 cohort differs from the original fixed comparison pool.")
    with np.load(study / "development_features.npz", allow_pickle=False) as a:
        for name, value in arrays.items():
            np.testing.assert_array_equal(value, a[name])
    config = {"experiment_id": "008_temporal_comparison", "conditions": list(CONDITIONS),
        "training": old_config, "source_study": str(study.resolve()),
        "original_checkpoint_sha256": previous["checkpoints_sha256"],
        "latency_warmup": 40, "latency_repeats": 200, "fresh_worker_processes": isolated}
    output = Path(output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    trials = []
    try:
        write_json(output / "status.json", {"state": "running", "completed_trials": 0})
        write_json(output / "config.json", config)
        write_json(output / "manifest.json", manifest)
        np.savez_compressed(output / "development_features.npz", **arrays)
        sources = {}
        for name in SOURCES:
            destination = output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, destination)
            sources[name] = sha256(destination)
        write_json(output / "environment.json", {"python": sys.version, "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(), "processor": platform.processor(),
            "device": "cpu", "source_sha256": sources,
            "input_sha256": {name: sha256(output / name) for name in
                             ("config.json", "manifest.json", "development_features.npz")}})
        print(f"Study folder: {output}", flush=True)
        for condition in CONDITIONS:
            for seed in old_config["seeds"]:
                folder = output / f"{condition}_seed_{seed}"
                folder.mkdir()
                write_json(folder / "job.json", {"condition": condition, "seed": seed})
                print(f"Fitting {condition}, seed {seed}...", flush=True)
                started = time.perf_counter()
                if isolated:
                    with (folder / "worker.log").open("w") as log:
                        subprocess.run([sys.executable, "-m", "research.experiments.temporal_comparison", "fit", str(folder)],
                                       cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
                else:
                    worker(folder)  # Synthetic integration tests only; RSS is not a fresh-process measurement.
                trial = fitting.read_json(folder / "metrics.json")
                trial["worker_wall_seconds"] = time.perf_counter() - started
                write_json(folder / "metrics.json", trial)
                trials.append(trial)
                write_json(output / "trials.json", trials)
                write_json(output / "status.json", {"state": "running", "completed_trials": len(trials)})
                if condition == "cnn":
                    prior = fitting.read_json(study / f"seed_{seed}/metrics.json")
                    for field in ("initial_state_sha256", "batch_indices_sha256", "best_step", "validation"):
                        if trial[field] != prior[field]:
                            raise ValueError(f"CNN reproduction failed before other architecture fits: {field}.")
                print(f"{condition} seed={seed}: accuracy={trial['validation']['accuracy']:.2%}, F1={trial['validation']['macro_f1']:.4f}, fit={trial['training_seconds']:.2f}s", flush=True)
        summary = summarize(trials, config)
        write_json(output / "summary.json", summary)
        (output / "report.md").write_text(report(summary))
        write_json(output / "verification.json", verify(output))
        write_json(output / "costs.json", {"new_human_labels": 0,
            "reused_training_labels": len(manifest["rows"]["train"]),
            "reused_development_labels": len(manifest["rows"]["validation"]),
            "fit_loop_seconds": sum(r["training_seconds"] for r in trials),
            "worker_wall_seconds": sum(r["worker_wall_seconds"] for r in trials),
            "scope": "Reusing already labeled data is not demonstrated annotation saving; historical costs remain."})
        write_json(output / "status.json", {"state": "complete", "completed_trials": len(trials)})
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "completed_trials": len(trials), "error": str(error)})
        raise
    print(f"All trials retained and verified: {output / 'report.md'}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    train = sub.add_parser("train")
    train.add_argument("--study", type=Path, default=STUDY)
    for action in ("fit", "verify"):
        sub.add_parser(action).add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.action == "train":
        run(args.study)
    elif args.action == "fit":
        worker(args.folder)
    else:
        print(verify(args.folder))


if __name__ == "__main__":
    main()
