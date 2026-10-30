"""Experiment 010: matched LSTM, explicit velocities and hand-reference ablations."""

import argparse
import copy
import hashlib
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import landmark_representations as rep
from . import temporal_comparison as previous
from .run import PROJECT, sha256, write_json

OUTPUT = PROJECT / "research/experiments/runs/motion_representation"
DEADLINES = {"p050": .5, "p100": 1.0, "p150": 1.5}
SOURCES = tuple(dict.fromkeys((*previous.SOURCES,
    "research/experiments/landmark_representations.py", "research/experiments/motion_representation.py",
    "research/experiments/test_motion_representation.py", "research/MOTION_REPRESENTATION_PROTOCOL.md")))


class Model(previous.RecurrentGesture):
    def __init__(self):
        super().__init__(bidirectional=False)
        self.recurrent = nn.LSTM(rep.FEATURES, 47, batch_first=True)
        self.head[2] = nn.Linear(47 * 4, 64)
        if sum(p.numel() for p in self.parameters()) != 49276:
            raise ValueError("Undeclared architecture change.")


def fit(arrays, seed, config):
    tx, ty = torch.from_numpy(arrays["train_x"]), torch.from_numpy(arrays["train_y"])
    vx, vy = torch.from_numpy(arrays["validation_x"]), torch.from_numpy(arrays["validation_y"])
    counts = np.bincount(arrays["train_y"], minlength=8)
    if np.any(counts < 5):
        raise ValueError("Need five training clips per class.")
    torch.manual_seed(seed)
    model = Model()
    initial = previous.fitting.state_hash(model)
    rng = torch.Generator().manual_seed(seed + 1_000_000)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (8 * counts), dtype=torch.float32))
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
                raise ValueError("Non-finite validation loss.")
            history.append({"step": step, "selection_loss": val_loss})
            if val_loss < best_loss:
                best_loss, best_state, best_step = val_loss, copy.deepcopy(model.state_dict()), step
    model.load_state_dict(best_state)
    model.eval()
    return model, history, {"seed": seed, "best_step": best_step,
        "training_seconds": time.perf_counter() - started, "initial_state_sha256": initial,
        "batch_indices_sha256": batch_hash.hexdigest(), "optimizer_steps": config["optimizer_steps"],
        "training_class_counts": counts.tolist(), "validation_checks": len(history)}


def snapshot(study):
    verified = previous.verify(study)
    _, manifest, _, original = previous.training.snapshot()
    if manifest != previous.fitting.read_json(Path(study) / "manifest.json"):
        raise ValueError("Original fixed cohort changed.")
    with np.load(Path(study) / "development_features.npz", allow_pickle=False) as saved:
        for name, value in original.items():
            np.testing.assert_array_equal(value, saved[name])
    arrays = {"train_y": original["train_y"], "validation_y": original["validation_y"]}
    by_file = {r["file"]: r for r in manifest["records"]}
    raw, audits, prefix_details = {}, {}, {}
    for row in manifest["records"]:
        if row["split"] not in ("train", "validation"):
            raise ValueError("No foreign data payload permitted.")
        if sha256(row["file"]) != row["sha256"]:
            raise ValueError("Recording changed.")
        xyz, times, visible, meta = previous.training.load_clip(row["file"], manifest["contract"])
        if any(meta[k] != row[k] for k in ("label", "split", "session")):
            raise ValueError("Recording role changed.")
        raw[row["file"]] = (xyz, times, visible, meta)
    for mode in rep.MODES:
        audits[mode] = {}
        for role in ("train", "validation"):
            features = []
            for file in manifest["rows"][role]:
                xyz, times, visible, meta = raw[file]
                x, audit = rep.features(xyz, times, visible, meta["aspect"], mode)
                features.append(x)
                audits[mode][file] = audit
            arrays[f"{mode}_{role}_x"] = np.stack(features)
        for tag, deadline in DEADLINES.items():
            features, details = [], []
            for file in manifest["rows"]["validation"]:
                xyz, times, visible, meta = raw[file]
                px, pt, pv = rep.prefix(xyz, times, visible, deadline)
                x, _ = rep.features(px, pt, pv, meta["aspect"], mode)
                features.append(x)
                details.append({"file": file, "frames": len(pt), "observed_seconds": float(pt[-1] - pt[0])})
            arrays[f"{mode}_{tag}_x"] = np.stack(features)
            prefix_details[tag] = details
    for file, row in by_file.items():
        if sha256(file) != row["sha256"]:
            raise ValueError("Raw source changed during feature extraction.")
    return manifest, arrays, audits, prefix_details, verified


def load_model(path):
    c = torch.load(path, map_location="cpu", weights_only=True)
    contract = c.get("command_contract", {})
    expected = {"experiment_id": "010_motion_representation", "feature_version": rep.VERSION,
        "features": rep.FEATURES, "labels": list(previous.v2.LABELS), "frames": rep.FRAMES,
        "architecture": "lstm146_hidden47_four_bins", "schema_version": 3}
    try:
        expected["command_contract"] = previous.training.collection.contract(contract["up_style"])
    except (KeyError, ValueError) as error:
        raise ValueError("Missing v3 contract.") from error
    if c.get("mode") not in rep.MODES or any(c.get(k) != value for k, value in expected.items()):
        raise ValueError("Incompatible experimental motion checkpoint.")
    model = Model()
    model.load_state_dict(c["state_dict"])
    model.eval()
    return model, c


def summarize(trials, config):
    seeds = config["training"]["seeds"]
    by = {(r["mode"], r["seed"]): r for r in trials}
    if len(by) != len(trials) or set(by) != {(m, s) for m in rep.MODES for s in seeds}:
        raise ValueError("Require all declared modes/seeds.")
    result = {"modes": {}, "paired_macro_f1_differences": {}}
    for mode in rep.MODES:
        rows = [by[mode, seed] for seed in seeds]
        result["modes"][mode] = {
            "validation": {field: {"mean": float(np.mean([r["validation"][field] for r in rows])),
                "seed_sd": float(np.std([r["validation"][field] for r in rows], ddof=1)) if len(rows) > 1 else None}
                for field in previous.fitting.FIELDS},
            "per_class_recall": {label: float(np.mean([r["validation"]["per_class"][label]["recall"] for r in rows])) for label in previous.v2.LABELS},
            "prefix_recorded_intent_agreement": {tag: float(np.mean([r["prefix_intent_metrics"][tag]["accuracy"] for r in rows])) for tag in DEADLINES},
            "training_seconds_mean": float(np.mean([r["training_seconds"] for r in rows])),
            "latency_median_ms_mean": float(np.mean([r["latency"]["median_ms"] for r in rows]))}
    for name, a, b in (("primary_camera_motion_minus_pose", "camera_motion", "camera_pose"),
                       ("hand_motion_minus_pose", "hand_motion", "hand_pose"),
                       ("hand_minus_camera_pose", "hand_pose", "camera_pose"),
                       ("hand_minus_camera_motion", "hand_motion", "camera_motion")):
        differences = [by[a, s]["validation"]["macro_f1"] - by[b, s]["validation"]["macro_f1"] for s in seeds]
        result["paired_macro_f1_differences"][name] = {"differences": differences, "mean": float(np.mean(differences))}
    result["scope"] = "Inspected single-user development; prefixes are original-intent agreement under distribution shift, not validated early accuracy or onset latency. No new algorithm/label savings/robot claim."
    return result


def verify(output):
    output = Path(output)
    config = previous.fitting.read_json(output / "config.json")
    manifest = previous.fitting.read_json(output / "manifest.json")
    env = previous.fitting.read_json(output / "environment.json")
    trials = previous.fitting.read_json(output / "trials.json")
    for name, digest in env["artifacts_sha256"].items():
        if sha256(output / name) != digest:
            raise ValueError("Archived experiment input/source changed.")
    for row in manifest["records"]:
        if sha256(row["file"]) != row["sha256"]:
            raise ValueError("Original recording changed.")
    torch.set_num_threads(config["training"]["threads"])
    states, batches, weights, vectors = {}, {}, {}, 0
    with np.load(output / "features.npz", allow_pickle=False) as arrays:
        for trial in trials:
            mode, seed = trial["mode"], trial["seed"]
            folder = output / f"{mode}_seed_{seed}"
            if sha256(folder / "model.pt") != trial["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed.")
            model, c = load_model(folder / "model.pt")
            expected = {"seed": seed, "mode": mode, "best_step": trial["best_step"],
                "manifest_sha256": sha256(output / "manifest.json"), "config_sha256": sha256(output / "config.json"),
                "command_contract": manifest["contract"], "training_files": manifest["rows"]["train"],
                "validation_files": manifest["rows"]["validation"]}
            if any(c.get(k) != value for k, value in expected.items()):
                raise ValueError("Checkpoint provenance changed.")
            for tag in ("full", *DEADLINES):
                x = arrays[f"{mode}_validation_x" if tag == "full" else f"{mode}_{tag}_x"]
                with torch.inference_mode():
                    p = model(torch.from_numpy(x)).softmax(1).numpy()
                rows = previous.fitting.read_json(folder / f"{tag}_predictions.json")
                if [r["file"] for r in rows] != manifest["rows"]["validation"] or [r["actual"] for r in rows] != [previous.v2.LABELS[int(i)] for i in arrays["validation_y"]]:
                    raise ValueError("Prediction row/intent labels changed.")
                np.testing.assert_allclose(p, [r["scores"] for r in rows], rtol=1e-6, atol=1e-7)
                if [r["predicted"] for r in rows] != [previous.v2.LABELS[int(i)] for i in p.argmax(1)]:
                    raise ValueError("Predicted name changed.")
                expected_metrics = trial["validation"] if tag == "full" else trial["prefix_intent_metrics"][tag]
                if previous.fitting.metric_set(p, arrays["validation_y"]) != expected_metrics:
                    raise ValueError("Saved metrics differ from reloaded weights.")
                vectors += len(rows)
            history = previous.fitting.read_json(folder / "history.json")
            chosen = min(history, key=lambda row: row["selection_loss"])
            if chosen["step"] != trial["best_step"] or len(history) != trial["validation_checks"] or not np.isclose(chosen["selection_loss"], trial["validation"]["nll"], rtol=1e-6, atol=1e-7):
                raise ValueError("Wrong minimum-loss checkpoint.")
            if previous.fitting.read_json(folder / "metrics.json") != trial:
                raise ValueError("Trial copies differ.")
            states.setdefault(seed, set()).add(trial["initial_state_sha256"])
            batches.setdefault(seed, set()).add(trial["batch_indices_sha256"])
            weights[str((folder / "model.pt").relative_to(output))] = trial["checkpoint_sha256"]
    if any(len(s) != 1 for s in (*states.values(), *batches.values())):
        raise ValueError("Initial weights or sample streams unpaired.")
    if previous.fitting.read_json(output / "summary.json") != summarize(trials, config):
        raise ValueError("Summary differs.")
    return {"input_files_verified": len(manifest["records"]), "prediction_vectors_verified": vectors,
            "all_initial_states_and_batch_streams_paired": True, "checkpoints_sha256": weights}


def run(study=previous.OUTPUT / "20261014T011524632211Z", output_root=OUTPUT):
    study = Path(study)
    config = {"experiment_id": "010_motion_representation", "training": previous.fitting.read_json(study / "config.json")["training"],
        "modes": list(rep.MODES), "prefix_deadlines_seconds": DEADLINES, "feature_version": rep.VERSION,
        "features": rep.FEATURES, "parameters": 49276, "source_study": str(study.resolve())}
    torch.set_num_threads(config["training"]["threads"])
    manifest, arrays, audits, prefixes, original_verified = snapshot(study)
    output = Path(output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    trials = []
    try:
        write_json(output / "status.json", {"state": "running", "completed_trials": 0})
        for name, value in (("config.json", config), ("manifest.json", manifest), ("feature_audits.json", audits),
                            ("prefix_observations.json", prefixes), ("original_verification.json", original_verified)):
            write_json(output / name, value)
        np.savez_compressed(output / "features.npz", **arrays)
        for name in SOURCES:
            path = output / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, path)
        pinned = [p for p in output.rglob("*") if p.is_file() and p.name != "status.json"]
        write_json(output / "environment.json", {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "device": "cpu", "artifacts_sha256": {str(p.relative_to(output)): sha256(p) for p in pinned}})
        print(f"Study folder: {output}", flush=True)
        for mode in rep.MODES:
            full_arrays = {"train_x": arrays[f"{mode}_train_x"], "validation_x": arrays[f"{mode}_validation_x"],
                           "train_y": arrays["train_y"], "validation_y": arrays["validation_y"]}
            for seed in config["training"]["seeds"]:
                folder = output / f"{mode}_seed_{seed}"
                folder.mkdir()
                model, history, stats = fit(full_arrays, seed, config["training"])
                torch.save({"state_dict": model.state_dict(), "experiment_id": config["experiment_id"],
                    "schema_version": 3, "feature_version": rep.VERSION, "features": rep.FEATURES, "frames": rep.FRAMES,
                    "labels": list(previous.v2.LABELS), "architecture": "lstm146_hidden47_four_bins", "mode": mode,
                    "seed": seed, "best_step": stats["best_step"], "command_contract": manifest["contract"],
                    "manifest_sha256": sha256(output / "manifest.json"), "config_sha256": sha256(output / "config.json"),
                    "training_files": manifest["rows"]["train"], "validation_files": manifest["rows"]["validation"]}, folder / "model.pt")
                write_json(folder / "history.json", history)
                model, _ = load_model(folder / "model.pt")
                metrics = {}
                for tag in ("full", *DEADLINES):
                    x = arrays[f"{mode}_validation_x" if tag == "full" else f"{mode}_{tag}_x"]
                    with torch.inference_mode():
                        p = model(torch.from_numpy(x)).softmax(1).numpy()
                    samples = list(zip(x, arrays["validation_y"], manifest["rows"]["validation"]))
                    write_json(folder / f"{tag}_predictions.json", previous.v2.predictions(samples, p))
                    metrics[tag] = previous.fitting.metric_set(p, arrays["validation_y"])
                trial = {**stats, "mode": mode, "validation": metrics.pop("full"), "prefix_intent_metrics": metrics,
                    "checkpoint_sha256": sha256(folder / "model.pt"),
                    "latency": previous.window_latency(model, torch.from_numpy(full_arrays["validation_x"][:1]), 40, 200)}
                write_json(folder / "metrics.json", trial)
                trials.append(trial)
                write_json(output / "trials.json", trials)
                write_json(output / "status.json", {"state": "running", "completed_trials": len(trials)})
                print(f"{mode} seed {seed}: accuracy={trial['validation']['accuracy']:.2%}, F1={trial['validation']['macro_f1']:.4f}, NONE→motion={trial['validation']['no_command_to_motion_rate']:.2%}", flush=True)
        summary = summarize(trials, config)
        write_json(output / "summary.json", summary)
        write_json(output / "verification.json", verify(output))
        previous.verify(study)
        write_json(output / "costs.json", {"new_human_labels": 0, "training_labels_reused": len(manifest["rows"]["train"]),
            "validation_labels_reused": len(manifest["rows"]["validation"]),
            "fit_loop_seconds": sum(r["training_seconds"] for r in trials),
            "scope": "Past collection/review effort and upstream pretraining not free; no demonstrated label saving."})
        write_json(output / "status.json", {"state": "complete", "completed_trials": len(trials)})
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "completed_trials": len(trials), "error": str(error)})
        raise
    print(f"All models and prefix outputs verified: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("train")
    sub.add_parser("verify").add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.action == "train":
        run()
    else:
        print(verify(args.folder))


if __name__ == "__main__":
    main()
