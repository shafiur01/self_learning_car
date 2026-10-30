"""Matched routine/targeted supervised studies; selection and monitoring separate.

Uses the prepared collection plan, existing v2 preprocessing/network and fixed
training recipe. No camera, test payloads, threshold tuning or app activation.
"""

import argparse
import copy
import hashlib
import json
import platform
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import boundary_pilot as pilot
from . import commands_v2 as v2
from .run import PROJECT, sha256, write_json

OUTPUT = PROJECT / "research/experiments/runs/boundary_data"
CONDITIONS = ("reference", "routine", "targeted")
ROLES = ("selection", "monitor")
FIELDS = ("accuracy", "macro_f1", "nll", "no_command_to_motion_rate",
          "no_command_to_down_rate", "stop_recall", "stop_to_motion_count",
          "forward_to_backward_count", "backward_to_forward_count")
SOURCES = (*v2.SOURCE_FILES, "research/experiments/boundary_pilot.py",
           "research/experiments/boundary_train.py", "research/experiments/test_boundary_train.py",
           "research/experiments/evaluate_v2_sessions.py", "research/experiments/run.py",
           "research/BOUNDARY_DATA_PLAN.md")


def read_json(path):
    return json.loads(Path(path).read_text())


def samples(records):
    rows = []
    for record in records:
        if sha256(record["file"]) != record["sha256"]:
            raise ValueError("Frozen input changed before feature extraction.")
        xyz, times, visible, meta = v2.load_clip(record["file"])
        if meta["label"] != record["label"] or meta["session"] != record["session"]:
            raise ValueError("Manifest and clip label/session differ.")
        rows.append((v2.features(xyz, times, visible, meta["aspect"]),
                     v2.LABELS.index(meta["label"]), record["file"]))
        if sha256(record["file"]) != record["sha256"]:
            raise ValueError("Input changed during feature extraction.")
    return rows


def snapshot(root):
    root = Path(root)
    # Reject foreign/test files from metadata before the collection auditor can
    # load their coordinates. The guided recorder only creates training files.
    for path in (root / "recordings").glob("*.npz"):
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(str(a["metadata"]))
        if meta.get("split") != "train" or meta.get("experiment_id") != pilot.EXPERIMENT:
            raise ValueError("Collection contains foreign or non-training data; no payload accessed.")
    plan, audited = pilot.read_plan(root), pilot.audit(root)
    if not audited["collection_complete"]:
        raise ValueError("Complete both sittings: need all 80 planned training clips.")
    config = plan["training_config"]
    if config != read_json(Path(plan["base_study"]) / "config.json"):
        raise ValueError("Comparison must reuse the reference training recipe and seeds.")
    original = read_json(Path(plan["base_study"]) / "manifest.json")["records"]
    by_name = {r["file"]: r for r in plan["base_records"]}
    roles = {"base_train": [r for r in original if r["split"] == "train"],
             "selection": [by_name[f] for f in plan["selection_validation_files"]],
             "monitor": [by_name[f] for f in plan["monitor_validation_files"]]}
    if any(r["split"] != "validation" for k in ROLES for r in roles[k]):
        raise ValueError("Selection/monitoring inputs must stay validation.")
    for group in pilot.GROUPS:
        extra = [r for r in audited["records"] if r["group"] == group]
        # Align the two conditions' appended row positions and batch-index streams.
        roles[group + "_add"] = sorted(extra, key=lambda r: (v2.LABELS.index(r["label"]),
                                                             r["sitting"], r["task_index"] % 5))
        if Counter(r["label"] for r in extra) != Counter({label: 10 for label in pilot.FOCUS}):
            raise ValueError("Added groups must be class matched: ten of each focus class.")
    assigned = [r["file"] for rows in roles.values() for r in rows]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(by_name) | {r["file"] for r in audited["records"]}:
        raise ValueError("Rows overlap or omit prepared development inputs.")
    batches = {key: samples(rows) for key, rows in roles.items()}
    for role in ("base_train", *ROLES):
        v2.require_classes(batches[role], role)
    arrays = {}
    for key, rows in batches.items():
        arrays[key + "_x"] = np.stack([r[0] for r in rows])
        arrays[key + "_y"] = np.asarray([r[1] for r in rows], dtype=np.int64)
    np.testing.assert_array_equal(arrays["routine_add_y"], arrays["targeted_add_y"])
    with np.load(Path(plan["base_study"]) / "development_features.npz", allow_pickle=False) as a:
        for source, target in (("train", "base_train"), ("validation", "selection")):
            for suffix in ("x", "y"):
                np.testing.assert_array_equal(arrays[f"{target}_{suffix}"], a[f"{source}_{suffix}"])
    monitoring = Path(plan["monitor_evaluation"])
    env = read_json(monitoring / "environment.json")
    for name, digest in env["artifact_sha256"].items():
        if sha256(monitoring / name) != digest:
            raise ValueError("Archived monitoring input changed.")
    if [r["file"] for r in read_json(monitoring / "manifest.json")["records"]] != plan["monitor_validation_files"]:
        raise ValueError("Monitoring order differs from the frozen collection plan.")
    with np.load(monitoring / "new_validation_features.npz", allow_pickle=False) as a:
        np.testing.assert_array_equal(arrays["monitor_x"], a["x"])
        np.testing.assert_array_equal(arrays["monitor_y"], a["y"])
    records = [{**r, "development_role": key} for key, rows in roles.items() for r in rows]
    attempts = [read_json(p) for p in sorted((root / "recording_attempts").glob("*.json"))]
    discarded = read_json(root/"discarded_effort.json") if (root/"discarded_effort.json").exists() else {}
    manifest = {"experiment_id": pilot.EXPERIMENT, "contract": v2.contract(), "records": records,
                "rows": {key: [r["file"] for r in rows] for key, rows in roles.items()},
                "summary": {key: len(rows) for key, rows in roles.items()},
                "tracking_quality_failures_new": audited["tracking_quality_failures"],
                "costs": {"usable_new_labels": len(audited["records"]), "new_labels_per_model": 40,
                          "attempt_logs": attempts, "logged_camera_wall_seconds": sum(a["wall_seconds"] for a in attempts),
                          "known_discarded_labels_before_restart": discarded.get("saved_labels_deleted"),
                          "limits": "Wall includes capture/replay/preparation/pauses, excludes deleted attempt and offline effort; not pure annotation time."},
                "scope": "Training and inspected development validation only; no test payloads."}
    return plan, config, manifest, batches, arrays


def metric_set(p, y):
    result = v2.metrics(p, y)
    cm = np.asarray(result["confusion_matrix_actual_rows_predicted_columns"])
    index = {label: i for i, label in enumerate(v2.LABELS)}
    none, down = index["NO_COMMAND"], index["DOWN"]
    result.update(no_command_to_down_count=int(cm[none, down]),
                  no_command_to_down_rate=float(cm[none, down] / cm[none].sum()),
                  forward_to_backward_count=int(cm[index["FORWARD"], index["BACKWARD"]]),
                  backward_to_forward_count=int(cm[index["BACKWARD"], index["FORWARD"]]))
    return result


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_candidate(train_x, train_y, selection_x, selection_y, seed, config):
    """Only training and checkpoint-selection data can enter this function."""
    tx, ty = torch.from_numpy(train_x), torch.from_numpy(train_y)
    vx, vy = torch.from_numpy(selection_x), torch.from_numpy(selection_y)
    counts = np.bincount(train_y, minlength=len(v2.LABELS))
    if np.any(counts < 5):
        raise ValueError("Need at least five training clips per class.")
    torch.manual_seed(seed)
    model = v2.build_model()
    initial_sha = state_hash(model)
    rng = torch.Generator().manual_seed(seed + 1_000_000)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    weights = torch.tensor(counts.sum() / (len(v2.LABELS)*counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    best_loss, best_state, best_step = float("inf"), None, None
    history, batch_hash = [], hashlib.sha256()
    started = time.perf_counter()
    for step in range(1, config["optimizer_steps"]+1):
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
    return model, history, {"seed": seed, "best_step": best_step, "training_seconds": time.perf_counter()-started,
           "optimizer_steps": config["optimizer_steps"], "validation_checks": len(history),
           "training_clips": len(tx), "training_class_counts": counts.tolist(),
           "initial_state_sha256": initial_sha, "batch_indices_sha256": batch_hash.hexdigest()}


def paired_summary(trials):
    by_key = {(r["condition"], r["seed"]): r for r in trials}
    seeds = sorted({r["seed"] for r in trials})
    if set(by_key) != {(c, seed) for c in CONDITIONS for seed in seeds} or len(trials) != len(by_key):
        raise ValueError("Require every declared condition/seed, without duplicates.")
    result = {"conditions": {c: {role: {} for role in ROLES} for c in CONDITIONS}}
    for condition in CONDITIONS:
        for role in ROLES:
            rows = [by_key[condition, seed][role] for seed in seeds]
            values = {field: {"mean": float(np.mean([r[field] for r in rows])),
                             "run_sd": float(np.std([r[field] for r in rows], ddof=1)) if len(rows)>1 else None}
                      for field in FIELDS}
            values["per_class"] = {label: {"support_per_model": rows[0]["per_class"][label]["support"],
                                          **{field: float(np.mean([r["per_class"][label][field] for r in rows]))
                                             for field in ("precision", "recall", "f1")}} for label in v2.LABELS}
            result["conditions"][condition][role] = values
    changes = {str(seed): by_key["targeted",seed]["monitor"]["macro_f1"]-
                           by_key["routine",seed]["monitor"]["macro_f1"] for seed in seeds}
    result["primary_comparison"] = {"field": "monitor macro F1, targeted minus routine",
                                    "paired_by_seed": changes, "mean": float(np.mean(list(changes.values())))}
    result["interpretation"] = ("Single-user supervised development comparison on inspected shared recordings; "
                                "no final test, novelty, label savings, SSL, unseen-user or robot reliability result.")
    return result


def report(summary, trials, manifest):
    lines = ["# Experiment 004: routine versus targeted additions", "",
             "Five-seed policy; selection and monitoring remain separate. All outcomes retained.", "",
             "| Condition | Selection accuracy | Monitor accuracy | Monitor macro F1 | Monitor NONE→motion | Monitor STOP recall |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for c in CONDITIONS:
        s, m = summary["conditions"][c]["selection"], summary["conditions"][c]["monitor"]
        lines.append(f"| {c} | {s['accuracy']['mean']:.2%} | {m['accuracy']['mean']:.2%} | "
                     f"{m['macro_f1']['mean']:.4f} | {m['no_command_to_motion_rate']['mean']:.2%} | {m['stop_recall']['mean']:.2%} |")
    lines += ["", f"Primary paired mean macro F1 difference: {summary['primary_comparison']['mean']:+.4f}.", "",
              "| Condition | Seed | Selection accuracy | Monitor accuracy | Monitor F1 | NONE→motion count | STOP→motion count | Saved step |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in trials:
        m = r["monitor"]
        lines.append(f"| {r['condition']} | {r['seed']} | {r['selection']['accuracy']:.2%} | {m['accuracy']:.2%} | "
                     f"{m['macro_f1']:.4f} | {m['no_command_to_motion_count']} | {m['stop_to_motion_count']} | {r['best_step']} |")
    lines += ["", "| Class | Reference recall | Routine recall | Targeted recall |",
              "| --- | ---: | ---: | ---: |"]
    for label in v2.LABELS:
        means = [summary["conditions"][c]["monitor"]["per_class"][label]["recall"] for c in CONDITIONS]
        lines.append(f"| {label} | " + " | ".join(f"{v:.2%}" for v in means) + " |")
    costs = manifest["costs"]
    lines += ["", f"New usable labels: {costs['usable_new_labels']}; per added model: {costs['new_labels_per_model']}.",
              f"Logged camera wall minutes: {costs['logged_camera_wall_seconds']/60:.2f}.", costs["limits"], "",
              summary["interpretation"], "Raw errors are not executed robot commands. Run SD is seed variability, not population uncertainty.", ""]
    return "\n".join(lines)


def verify(output):
    output = Path(output)
    env, manifest, trials = [read_json(output/name) for name in ("environment.json", "manifest.json", "trials.json")]
    for name, digest in env["input_sha256"].items():
        if sha256(output/name) != digest:
            raise ValueError("Archived study input changed.")
    for name, digest in env["source_sha256"].items():
        if sha256(output/"source"/name) != digest:
            raise ValueError("Archived source changed.")
    for r in manifest["records"]:
        if sha256(r["file"]) != r["sha256"]:
            raise ValueError("Original recording changed during the study.")
    with np.load(output/"development_features.npz", allow_pickle=False) as a:
        arrays = {k: a[k].copy() for k in a.files}
    verified, vectors = {}, 0
    for r in trials:
        folder = output/r["condition"]/f"seed_{r['seed']}"
        if sha256(folder/"model.pt") != r["checkpoint_sha256"]:
            raise ValueError("Candidate changed after evaluation.")
        model, checkpoint = v2.load_model(folder/"model.pt")
        if r["condition"] != "reference" and (checkpoint["manifest_sha256"] != sha256(output/"manifest.json")
              or checkpoint["training_files"] != manifest["rows"]["base_train"]+manifest["rows"][r["condition"]+"_add"]
              or checkpoint["selection_files"] != manifest["rows"]["selection"]):
            raise ValueError("Candidate training/selection provenance differs.")
        for role in ROLES:
            rows = read_json(folder/f"{role}_predictions.json")
            if [p["file"] for p in rows] != manifest["rows"][role]:
                raise ValueError("Saved prediction rows differ from frozen inputs.")
            with torch.inference_mode():
                p = model(torch.from_numpy(arrays[role+"_x"])).softmax(1).numpy()
            np.testing.assert_allclose(p, [v["scores"] for v in rows], rtol=1e-6, atol=1e-7)
            if metric_set(p, arrays[role+"_y"]) != r[role]:
                raise ValueError("Saved metrics differ from checkpoint predictions.")
            vectors += len(rows)
        verified[str((folder/"model.pt").relative_to(output))] = r["checkpoint_sha256"]
    by_key = {(r["condition"],r["seed"]): r for r in trials}
    for seed in {r["seed"] for r in trials}:
        for field in ("initial_state_sha256", "batch_indices_sha256", "training_class_counts"):
            if by_key["routine",seed][field] != by_key["targeted",seed][field]:
                raise ValueError("Matched initial weights, sampling or class counts differ.")
    if read_json(output/"summary.json") != paired_summary(trials):
        raise ValueError("Summary does not reproduce all archived trials.")
    return {"reference_and_new_input_files_verified": len(manifest["records"]),
            "saved_prediction_vectors_verified": vectors, "checkpoints_sha256": verified,
            "paired_initial_states_batch_indices_class_counts_verified": True,
            "scope": "Artifact/software integrity, not population or robot reliability."}


def run(root=pilot.ROOT, output_root=OUTPUT):
    plan, config, manifest, batches, arrays = snapshot(root)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    output = Path(output_root)/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    trials = []
    write_json(output/"status.json", {"state": "preparing", "trained_trials": 0, "evaluated_candidates": 0})
    try:
        write_json(output/"config.json", {"training": config, "conditions": list(CONDITIONS),
                   "primary_comparison": plan["primary_comparison"], "selection": "original validation only",
                   "monitor": "inspected second validation; no checkpoint selection or final-test claim"})
        write_json(output/"manifest.json", manifest)
        write_json(output/"contract.json", v2.contract())
        shutil.copyfile(Path(root)/"plan.json", output/"collection_plan.json")
        np.savez_compressed(output/"development_features.npz", **arrays)
        source_hashes = {}
        for name in SOURCES:
            target = output/"source"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT/name, target)
            source_hashes[name] = sha256(target)
        write_json(output/"environment.json", {"python": sys.version, "torch": torch.__version__,
                   "numpy": np.__version__, "platform": platform.platform(), "source_sha256": source_hashes,
                   "input_sha256": {name: sha256(output/name) for name in
                                     ("config.json", "manifest.json", "contract.json", "collection_plan.json", "development_features.npz")}})
        print(f"Study folder: {output}", flush=True)
        for condition in CONDITIONS:
            for seed in config["seeds"]:
                folder = output/condition/f"seed_{seed}"
                folder.mkdir(parents=True)
                if condition == "reference":
                    candidate = next(r for r in plan["base_checkpoints"] if r["seed"] == seed)
                    original = Path(plan["base_study"])/candidate["file"]
                    shutil.copyfile(original, folder/"model.pt")
                    if sha256(folder/"model.pt") != candidate["sha256"]:
                        raise ValueError("Reference checkpoint copy differs.")
                    stats = {"seed": seed, "training_clips": len(arrays["base_train_y"]), "training_seconds": 0,
                             "best_step": read_json(Path(plan["base_study"])/f"seed_{seed}/metrics.json")["best_step"]}
                else:
                    tx = np.concatenate((arrays["base_train_x"], arrays[condition+"_add_x"]))
                    ty = np.concatenate((arrays["base_train_y"], arrays[condition+"_add_y"]))
                    model, history, stats = fit_candidate(tx, ty, arrays["selection_x"], arrays["selection_y"], seed, config)
                    torch.save({"state_dict": model.state_dict(), "schema_version": v2.VERSION,
                                "feature_version": 1, "taxonomy": v2.TAXONOMY, "labels": list(v2.LABELS),
                                "convention": v2.CONVENTION, "frames": v2.FRAMES, "seconds": v2.SECONDS,
                                "command_contract": v2.contract(), "seed": seed, "condition": condition,
                                "best_step": stats["best_step"], "manifest_sha256": sha256(output/"manifest.json"),
                                "development_sessions": sorted({r["session"] for r in manifest["records"]}),
                                "development_files": [r["file"] for r in manifest["records"]],
                                "training_files": manifest["rows"]["base_train"]+manifest["rows"][condition+"_add"],
                                "selection_files": manifest["rows"]["selection"],
                                "monitor_files": manifest["rows"]["monitor"]}, folder/"model.pt")
                    write_json(folder/"history.json", history)
                model, _ = v2.load_model(folder/"model.pt")  # Evaluate exactly the saved weights.
                result = {"condition": condition, **stats, "checkpoint_sha256": sha256(folder/"model.pt")}
                for role in ROLES:
                    with torch.inference_mode():
                        p = model(torch.from_numpy(arrays[role+"_x"])).softmax(1).numpy()
                    result[role] = metric_set(p, arrays[role+"_y"])
                    write_json(folder/f"{role}_predictions.json", v2.predictions(batches[role], p))
                    if condition == "reference":
                        original = (Path(plan["base_study"])/f"seed_{seed}/validation_predictions.json" if role == "selection"
                                    else Path(plan["monitor_evaluation"])/f"seed_{seed}/predictions.json")
                        saved = read_json(original)
                        if [r["file"] for r in saved] != manifest["rows"][role]:
                            raise ValueError("Reference prediction order changed.")
                        np.testing.assert_allclose(p, [r["scores"] for r in saved], rtol=1e-6, atol=1e-7)
                write_json(folder/"metrics.json", result)
                trials.append(result)
                write_json(output/"trials.json", trials)
                write_json(output/"status.json", {"state": "running", "trained_trials": sum(r["condition"]!="reference" for r in trials),
                                                   "evaluated_candidates": len(trials)})
                print(f"{condition} seed={seed}: monitor accuracy={result['monitor']['accuracy']:.2%}, F1={result['monitor']['macro_f1']:.4f}", flush=True)
        summary = paired_summary(trials)
        write_json(output/"summary.json", summary)
        (output/"report.md").write_text(report(summary, trials, manifest))
        verification = verify(output)
        pilot.read_plan(root)  # Recheck original weights/source/inputs, not just copied candidates.
        write_json(output/"verification.json", verification)
        write_json(output/"status.json", {"state": "complete", "trained_trials": 2*len(config["seeds"]),
                                          "evaluated_candidates": len(trials)})
    except BaseException as error:
        write_json(output/"status.json", {"state": "failed", "trained_trials": sum(r["condition"]!="reference" for r in trials),
                                          "evaluated_candidates": len(trials), "error": str(error)})
        raise
    print(f"Study verified. Read {output/'report.md'}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=pilot.ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        run(args.data, args.output)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
