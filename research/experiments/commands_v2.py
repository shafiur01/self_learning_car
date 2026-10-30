"""Versioned eight-command data, recording, training, and frozen test workflow.

Reuses the legacy landmark features and temporal CNN. Original data/weights
are never modified and no candidate is automatically activated in the camera.
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
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import torch
from torch import nn

from gesture_commands.data import (CONVENTION, FEATURES, FRAMES, LABELS as OLD_LABELS,
                                   SECONDS, features, load_clip as load_old_clip,
                                   quality, validate_clip)
from gesture_commands.model import GestureNet
from gesture_commands.train import tensors
from .run import PROJECT, sha256, write_json

LABELS = (*OLD_LABELS, "UP", "DOWN")
KEYS = dict(zip("fblrsnud", LABELS))
MOVEMENT = frozenset(("FORWARD", "BACKWARD", "LEFT", "RIGHT", "UP", "DOWN"))
TAXONOMY = "commands_v2"
VERSION = 2
DATA = PROJECT / "research/experiments/data/commands_v2"
OUTPUT = PROJECT / "research/experiments/runs/commands_v2"
STUDY = PROJECT / "research/experiments/runs/20261008T005037258990Z"
NEW_EVALUATION = STUDY / "session_evaluations/20261008T013739665513Z"
SOURCE_FILES = ("research/experiments/commands_v2.py", "gesture_commands/app.py",
                "gesture_commands/data.py", "gesture_commands/model.py",
                "gesture_commands/train.py", "research/PROTOCOL.md")
TRAIN_CONFIG = {"seeds": [11, 22, 33, 44, 55], "optimizer_steps": 560,
                "validation_every_steps": 20, "batch_size": 32,
                "learning_rate": 0.001, "threads": 2, "augmentation": "none"}
MEANINGS = {
    "FORWARD": "Beckon toward yourself using one, several, or all fingers.",
    "BACKWARD": "Push your hand outward to signal go away.",
    "LEFT": "Point toward the left side of the mirrored preview.",
    "RIGHT": "Point toward the right side of the mirrored preview.",
    "STOP": "Hold a closed fist still, distinct from beckoning.",
    "NO_COMMAND": "No intended command: resting, unrelated motion, transitions, or no hand.",
    "UP": "Deliberately point upward or move the hand upward to request UP; varied styles allowed.",
    "DOWN": "Deliberately point downward or move the hand downward to request DOWN; varied styles allowed.",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def contract():
    return {"taxonomy": TAXONOMY, "schema_version": VERSION, "feature_version": 1,
            "labels": list(LABELS), "convention": CONVENTION, "meanings": MEANINGS,
            "vertical_styles": ["pointing", "whole_hand_motion"],
            "robot_mapping": "unassigned; screen-only labels; controller-specific actions"}


def require_contract(folder):
    value = json.loads((Path(folder) / "contract.json").read_text())
    if value != contract():
        raise ValueError("Dataset command contract differs; use a separately versioned study.")
    return value


def content_sha(xyz, times, visible, aspect):
    identity = hashlib.sha256()
    for a in (xyz, times - times[0], visible, np.asarray([aspect], dtype=np.float64)):
        identity.update(a.tobytes())
    return identity.hexdigest()


def save_clip(folder, xyz, times, visible, metadata, filename=None):
    xyz, times, visible = validate_clip(xyz, times, visible)
    if metadata["label"] not in LABELS or metadata["split"] not in ("train", "validation", "test"):
        raise ValueError("Unknown v2 label/split.")
    if not isinstance(metadata.get("session"), str) or not metadata["session"]:
        raise ValueError("A recording session is required.")
    aspect = metadata["aspect"]
    if not np.isfinite(aspect) or aspect <= 0:
        raise ValueError("Invalid camera aspect ratio.")
    meta = {**metadata, "schema_version": VERSION, "feature_version": 1,
            "taxonomy": TAXONOMY, "convention": CONVENTION, "labels": list(LABELS)}
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (filename or f"{uuid4().hex}.npz")
    if path.exists():
        raise ValueError(f"Refusing to overwrite recording: {path}")
    with path.open("xb") as stream:
        np.savez_compressed(stream, xyz=xyz, times=times-times[0], visible=visible,
                            metadata=json.dumps(meta))
    return path


def load_clip(path):
    with np.load(path, allow_pickle=False) as a:
        xyz, times, visible = validate_clip(a["xyz"], a["times"], a["visible"])
        meta = json.loads(str(a["metadata"]))
    expected = {"schema_version": VERSION, "feature_version": 1,
                "taxonomy": TAXONOMY, "labels": list(LABELS), "convention": CONVENTION}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Recording has an incompatible v2 schema: {path}")
    if meta.get("label") not in LABELS or meta.get("split") not in ("train", "validation", "test"):
        raise ValueError(f"Invalid label/split: {path}")
    if not np.isfinite(meta["aspect"]) or meta["aspect"] <= 0:
        raise ValueError(f"Invalid aspect: {path}")
    return xyz, times, visible, meta


def prepare(folder=DATA, study=STUDY, evaluation=NEW_EVALUATION):
    """Copy whole authorized cohorts; retain old validation roles in provenance."""
    folder, study, evaluation = map(Path, (folder, study, evaluation))
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "migration.json").exists():
        require_contract(folder)
        migration = json.loads((folder / "migration.json").read_text())
        for r in migration["imports"]:
            if sha256(folder / r["destination"]) != r["destination_sha256"]:
                raise ValueError("Previously imported v2 recording changed.")
            if sha256(PROJECT / r["origin_file"]) != r["origin_sha256"]:
                raise ValueError("Original recording changed after migration.")
        print(f"Already prepared: {len(migration['imports'])} preserved imports in {folder}")
        return migration
    old = json.loads((study / "manifest.json").read_text())
    fresh = json.loads((evaluation / "manifest.json").read_text())
    # This is the exact previously evaluated cohort, not arbitrary new files.
    if any(r["split"] != "validation" for r in fresh["records"]):
        raise ValueError("The authorized recycled cohort must be validation only.")
    if {r["file"] for r in fresh["records"]}.intersection(r["file"] for r in old["records"]):
        raise ValueError("Recycled cohort overlaps the original study manifest.")
    selected = [r for r in old["records"] if r["split"] == "train"] + fresh["records"]
    sessions = {r["session"] for r in selected}
    retired = [r for r in old["records"] if r["split"] == "validation"]
    if sessions.intersection(r["session"] for r in retired):
        raise ValueError("Training migration would split a retired session.")
    # Test metadata checks old session identity; test payloads/labels are excluded.
    historical_sessions = {r["session"] for r in (*old["records"], *fresh["records"])}
    for path in (PROJECT / "gesture_commands/data").glob("*.npz"):
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(str(a["metadata"]))
        historical_sessions.add(meta["session"])
    write_json(folder / "contract.json", contract())
    imports = []
    write_json(folder / "preparation_status.json", {"state": "copying", "imports": 0})
    try:
        for record in selected:
            source = PROJECT / record["file"]
            if sha256(source) != record["sha256"]:
                raise ValueError(f"Source recording changed: {source}")
            xyz, times, visible, meta = load_old_clip(source)
            if any(meta.get(k) != record[k] for k in ("label", "split", "session")):
                raise ValueError("Frozen manifest and original metadata differ.")
            recycled = record["split"] == "validation"
            target_meta = {"label": meta["label"], "split": "train", "session": meta["session"],
                           "aspect": meta["aspect"], "source": meta.get("source", "unknown"),
                           "origin_file": record["file"], "origin_sha256": record["sha256"],
                           "original_metadata": meta, "origin_split": meta["split"],
                           "acquisition": "recycled_inspected_validation" if recycled else "legacy_training",
                           "prediction": None, "scores": None, "migration_utc": utc_now()}
            destination = save_clip(folder, xyz, times, visible, target_meta, filename=source.name)
            if sha256(source) != record["sha256"]:
                raise ValueError("Source changed during copying.")
            imports.append({"destination": destination.name, "destination_sha256": sha256(destination),
                            "origin_file": record["file"], "origin_sha256": record["sha256"],
                            "origin_split": record["split"], "role": "train", "session": record["session"],
                            "label": record["label"], "source": record["source"],
                            "recycled_validation": recycled})
        migration = {"created_utc": utc_now(), "authorization": "User requested recycling the new validation session for training.",
                     "imports": imports, "training_labels": dict(Counter(r["label"] for r in imports)),
                     "recycled_sessions": sorted({r["session"] for r in fresh["records"]}),
                     "historical_sessions": sorted(historical_sessions),
                     "historical_development_raw_hashes": sorted({r["raw_content_sha256"] for r in (*old["records"], *fresh["records"])}),
                     "retired_legacy_validation_files": [r["file"] for r in retired],
                     "source_manifest_sha256": sha256(study / "manifest.json"),
                     "recycled_manifest_sha256": sha256(evaluation / "manifest.json"),
                     "notes": "All originals unchanged. No legacy validation/test enters fresh v2 evaluation. Corrections retain provenance; no low-label cost claim."}
        write_json(folder / "migration.json", migration)
        write_json(folder / "preparation_status.json", {"state": "complete", "imports": len(imports)})
    except BaseException as error:
        write_json(folder / "preparation_status.json", {"state": "failed", "imports": len(imports), "error": str(error)})
        raise
    print(f"Prepared {len(imports)} training copies in {folder}; fresh validation/test: zero.")
    return migration


def build_model():
    model = GestureNet()
    model.network[-1] = nn.Linear(model.network[-1].in_features, len(LABELS))
    return model


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"schema_version": VERSION, "feature_version": 1, "taxonomy": TAXONOMY,
                "labels": list(LABELS), "convention": CONVENTION, "frames": FRAMES,
                "seconds": SECONDS, "command_contract": contract()}
    if any(checkpoint.get(k) != value for k, value in expected.items()):
        raise ValueError("Checkpoint is not a compatible eight-command v2 model.")
    model = build_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


class CommandGate:
    """Same demo thresholds, with explicit six movement labels rather than indices."""

    def __init__(self, threshold=0.8, margin=0.2, repeats=3):
        self.threshold, self.margin, self.repeats = threshold, margin, repeats
        self.reset()

    def reset(self):
        self.candidate, self.count = None, 0

    def update(self, probabilities, tracked):
        if not tracked:
            self.reset()
            return "NO_HAND", "STOP"
        p = np.asarray(probabilities)
        if p.shape != (len(LABELS),) or not np.isfinite(p).all():
            raise ValueError("Eight finite class scores are required.")
        order = np.argsort(p)
        if p[order[-1]] < self.threshold or p[order[-1]] - p[order[-2]] < self.margin:
            self.reset()
            return "UNCERTAIN", "STOP"
        label = LABELS[order[-1]]
        if label not in MOVEMENT:
            self.reset()
            return label, "STOP"
        self.count = self.count + 1 if self.candidate == label else 1
        self.candidate = label
        return label, label if self.count >= self.repeats else "STOP"


def inventory(folder):
    require_contract(folder)
    counts = {s: Counter({label: 0 for label in LABELS}) for s in ("train", "validation", "test")}
    sessions = {s: set() for s in counts}
    roles = {}
    for path in sorted(Path(folder).glob("*.npz")):
        with np.load(path, allow_pickle=False) as a:
            m = json.loads(str(a["metadata"]))
        split, session = m["split"], m["session"]
        if split not in counts or m["label"] not in LABELS or not session:
            raise ValueError("Invalid recording metadata.")
        if session in roles and roles[session] != split:
            raise ValueError("One session cannot cross splits.")
        roles[session] = split
        counts[split][m["label"]] += 1
        sessions[split].add(session)
    result = {split: {"clips": sum(c.values()), "labels": dict(c), "sessions": sorted(sessions[split])}
              for split, c in counts.items()}
    print(json.dumps(result, indent=2))
    return result


def development_snapshot(folder):
    require_contract(folder)
    migration = json.loads((Path(folder) / "migration.json").read_text())
    historical = set(migration["historical_sessions"])
    old_contents = set(migration["historical_development_raw_hashes"])
    for record in migration["imports"]:
        if sha256(Path(folder) / record["destination"]) != record["destination_sha256"]:
            raise ValueError("An imported recording changed; preserve the migration and create an explicit new version.")
    dataset = {"train": [], "validation": []}
    records, roles, contents = [], {}, {}
    for path in sorted(Path(folder).glob("*.npz")):
        with np.load(path, allow_pickle=False) as a:
            m = json.loads(str(a["metadata"]))
        split, session = m["split"], m["session"]
        if split not in ("train", "validation", "test") or not isinstance(session, str) or not session:
            raise ValueError("Invalid split/session.")
        if session in roles and roles[session] != split:
            raise ValueError("Session crosses splits.")
        roles[session] = split
        if split == "test":
            continue  # Do not read test arrays or labels during training.
        digest = sha256(path)
        xyz, times, visible, meta = load_clip(path)
        identity = content_sha(xyz, times, visible, meta["aspect"])
        if split == "validation":
            if session in historical or identity in old_contents or meta.get("source") != "recording":
                raise ValueError("V2 validation must be fresh original sessions/contents, not legacy or correction clips.")
        if identity in contents:
            raise ValueError("Duplicate raw contents in development data.")
        contents[identity] = split
        if sha256(path) != digest:
            raise ValueError("Recording changed during reading.")
        name = str(path.resolve())
        dataset[split].append((features(xyz, times, visible, meta["aspect"]), LABELS.index(meta["label"]), name))
        records.append({"file": name, "sha256": digest, "raw_content_sha256": identity,
                        "split": split, "session": session, "label": meta["label"],
                        "source": meta.get("source", "unknown"), "origin_split": meta.get("origin_split"),
                        "acquisition": meta.get("acquisition", "fresh_recording"),
                        "passes_tracking_quality": quality(times, visible)})
    return dataset, {"contract": contract(), "records": records, "scope": "train/validation; test payloads excluded"}


def require_classes(samples, split, minimum=5):
    counts = np.bincount([s[1] for s in samples], minlength=len(LABELS))
    missing = [f"{label}: {count}/{minimum}" for label, count in zip(LABELS, counts) if count < minimum]
    if missing:
        raise ValueError(f"Need more {split} recordings: " + ", ".join(missing))
    return counts


def metrics(probabilities, targets):
    predicted = probabilities.argmax(1)
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(cm, (targets, predicted), 1)
    tp = np.diag(cm)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(len(LABELS)), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(len(LABELS)), where=cm.sum(1) != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(len(LABELS)), where=precision+recall != 0)
    motion = np.isin(predicted, [LABELS.index(label) for label in MOVEMENT])
    none = targets == LABELS.index("NO_COMMAND")
    stop = targets == LABELS.index("STOP")
    return {"clips": len(targets), "labels": list(LABELS), "accuracy": float(np.mean(predicted == targets)),
            "macro_f1": float(np.mean(f1)), "nll": float(-np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-12)).mean()),
            "no_command_to_motion_count": int(np.sum(none & motion)),
            "no_command_to_motion_rate": float(np.sum(none & motion) / none.sum()) if none.any() else None,
            "stop_to_motion_count": int(np.sum(stop & motion)),
            "stop_recall": float(recall[LABELS.index("STOP")]) if stop.any() else None,
            "confusion_matrix_actual_rows_predicted_columns": cm.tolist(),
            "per_class": {label: {"support": int(cm[i].sum()), "precision": float(precision[i]),
                                   "recall": float(recall[i]), "f1": float(f1[i])} for i, label in enumerate(LABELS)}}


def train(folder=DATA, output_root=OUTPUT, config=None):
    config = dict(TRAIN_CONFIG if config is None else config)
    if (not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"])
            or any(type(s) is not int or s < 0 for s in config["seeds"])
            or any(type(config[k]) is not int or config[k] < 1 for k in
                   ("optimizer_steps", "validation_every_steps", "batch_size", "threads"))
            or not np.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0
            or config["augmentation"] != "none"):
        raise ValueError("Invalid fixed-update training configuration.")
    dataset, manifest = development_snapshot(folder)
    counts = require_classes(dataset["train"], "train")
    require_classes(dataset["validation"], "validation")
    # All validation sessions include every class for a meaningful session report.
    for session in {r["session"] for r in manifest["records"] if r["split"] == "validation"}:
        if {r["label"] for r in manifest["records"] if r["session"] == session} != set(LABELS):
            raise ValueError("Each fresh validation session must cover all eight labels.")
    output = Path(output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"state": "preparing", "completed_trials": 0})
    results = []
    try:
        write_json(output / "config.json", config)
        write_json(output / "manifest.json", manifest)
        write_json(output / "contract.json", contract())
        tx, ty = tensors(dataset["train"])
        vx, vy = tensors(dataset["validation"])
        np.savez_compressed(output / "development_features.npz", train_x=tx.numpy(), train_y=ty.numpy(),
                            validation_x=vx.numpy(), validation_y=vy.numpy())
        source_hashes = {}
        for name in SOURCE_FILES:
            destination = output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, destination)
            source_hashes[name] = sha256(destination)
        write_json(output / "environment.json", {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                   "platform": platform.platform(), "source_sha256": source_hashes,
                   "input_sha256": {name: sha256(output / name) for name in
                                    ("config.json", "manifest.json", "contract.json", "development_features.npz")}})
        torch.set_num_threads(config["threads"])
        torch.use_deterministic_algorithms(True)
        for seed in config["seeds"]:
            torch.manual_seed(seed)
            model = build_model()
            rng = torch.Generator().manual_seed(seed + 1_000_000)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(LABELS) * counts), dtype=torch.float32))
            best_loss, best_state, best_step = float("inf"), None, None
            history = []
            started = time.perf_counter()
            for step in range(1, config["optimizer_steps"]+1):
                model.train()
                indices = torch.randint(len(tx), (config["batch_size"],), generator=rng)
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
                    history.append({"step": step, "validation_loss": val_loss})
                    if val_loss < best_loss:
                        best_loss, best_state, best_step = val_loss, copy.deepcopy(model.state_dict()), step
            model.load_state_dict(best_state)
            model.eval()
            with torch.inference_mode():
                p = model(vx).softmax(1).numpy()
            result = metrics(p, vy.numpy())
            result.update(seed=seed, best_step=best_step, training_seconds=time.perf_counter()-started,
                          training_clips=len(tx), optimizer_steps=config["optimizer_steps"],
                          validation_checks=len(history))
            trial = output / f"seed_{seed}"
            trial.mkdir()
            checkpoint = {"state_dict": model.state_dict(), "schema_version": VERSION, "feature_version": 1,
                          "taxonomy": TAXONOMY, "labels": list(LABELS), "frames": FRAMES, "seconds": SECONDS,
                          "convention": CONVENTION, "command_contract": contract(), "seed": seed, "best_step": best_step,
                          "development_sessions": sorted({r["session"] for r in manifest["records"]}),
                          "development_files": [r["file"] for r in manifest["records"]],
                          "manifest_sha256": sha256(output / "manifest.json")}
            torch.save(checkpoint, trial / "model.pt")
            write_json(trial / "metrics.json", result)
            write_json(trial / "history.json", history)
            write_json(trial / "validation_predictions.json", predictions(dataset["validation"], p))
            by_session = {}
            for session in sorted({r["session"] for r in manifest["records"] if r["split"] == "validation"}):
                names = {r["file"] for r in manifest["records"] if r["session"] == session}
                indices = [i for i, s in enumerate(dataset["validation"]) if s[2] in names]
                by_session[session] = metrics(p[indices], vy.numpy()[indices])
            write_json(trial / "session_metrics.json", by_session)
            results.append(result)
            write_json(output / "trials.json", results)
            write_json(output / "status.json", {"state": "running", "completed_trials": len(results)})
            print(f"seed={seed}: validation accuracy={result['accuracy']:.2%}, F1={result['macro_f1']:.4f}", flush=True)
        for r in manifest["records"]:
            if sha256(r["file"]) != r["sha256"]:
                raise ValueError("Development input changed during training.")
        write_json(output / "summary.json", summarize(results))
        write_json(output / "status.json", {"state": "complete", "completed_trials": len(results)})
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "completed_trials": len(results), "error": str(error)})
        raise
    print(f"Saved all candidates: {output}. No camera model was activated.")
    return output


def predictions(samples, probabilities):
    return [{"file": s[2], "actual": LABELS[s[1]], "predicted": LABELS[int(p.argmax())], "scores": p.tolist()}
            for s, p in zip(samples, probabilities)]


def summarize(results):
    return {"trials": results, "metrics": {field: {"mean": float(np.mean([r[field] for r in results])),
                  "run_sd": float(np.std([r[field] for r in results], ddof=1)) if len(results) > 1 else None}
                  for field in ("accuracy", "macro_f1", "nll", "no_command_to_motion_rate", "stop_recall")},
            "interpretation": "Eight-class single-user supervised pilot; all seeds share recordings. "
                              "No evidence of label savings, stage-order advantage, or physical robot reliability."}


def freeze(run):
    """Declare all candidates/choices fixed before collecting final test sessions."""
    run = Path(run)
    if (run / "freeze.json").exists():
        raise ValueError("This study is already frozen; preserve its freeze record.")
    config = json.loads((run / "config.json").read_text())
    if json.loads((run / "status.json").read_text()) != {"state": "complete", "completed_trials": len(config["seeds"])}:
        raise ValueError("Freeze requires a complete study.")
    env = json.loads((run / "environment.json").read_text())
    for name, digest in env["input_sha256"].items():
        if sha256(run / name) != digest:
            raise ValueError("Frozen-study input changed.")
    for seed in config["seeds"]:
        _, checkpoint = load_model(run / f"seed_{seed}/model.pt")
        if checkpoint["seed"] != seed or checkpoint["manifest_sha256"] != sha256(run / "manifest.json"):
            raise ValueError("Checkpoint provenance differs from the completed study.")
    write_json(run / "freeze.json", {"created_utc": utc_now(), "taxonomy": TAXONOMY,
               "checkpoints": [{"seed": seed, "file": f"seed_{seed}/model.pt", "sha256": sha256(run / f"seed_{seed}/model.pt")}
                               for seed in config["seeds"]],
               "input_sha256": env["input_sha256"], "source_sha256": env["source_sha256"],
               "rule": "All candidates, raw eight-class classification, no ensemble or tuning on test."})
    print("Study frozen. Collect genuinely new test sessions now; all declared candidates will be evaluated.")


def evaluate_test(run, folder=DATA):
    run, folder = Path(run), Path(folder)
    require_contract(folder)
    frozen = json.loads((run / "freeze.json").read_text())
    for name, digest in frozen["input_sha256"].items():
        if sha256(run / name) != digest:
            raise ValueError("Frozen study input changed.")
    for name in ("research/experiments/commands_v2.py", "gesture_commands/data.py", "gesture_commands/model.py"):
        if sha256(PROJECT / name) != frozen["source_sha256"][name]:
            raise ValueError("Frozen evaluation code/features changed.")
    for r in frozen["checkpoints"]:
        if sha256(run / r["file"]) != r["sha256"]:
            raise ValueError("Frozen checkpoint changed.")
    migration = json.loads((folder / "migration.json").read_text())
    development = json.loads((run / "manifest.json").read_text())
    banned_sessions = set(migration["historical_sessions"]) | {r["session"] for r in development["records"]}
    banned_contents = set(migration["historical_development_raw_hashes"]) | {r["raw_content_sha256"] for r in development["records"]}
    ledger_path = folder / "test_access.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"accesses": []}
    seen_sessions = {session for access in ledger["accesses"] for session in access["sessions"]}
    banned_contents.update(identity for access in ledger["accesses"] for identity in access.get("raw_content_sha256", []))
    candidates, roles, candidate_sessions = [], {}, set()
    test_counts, session_labels = Counter(), {}
    for path in sorted(folder.glob("*.npz")):
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(str(a["metadata"]))
        split, session = meta["split"], meta["session"]
        if session in roles and roles[session] != split:
            raise ValueError("Session crosses splits.")
        roles[session] = split
        if split != "test":
            continue
        if session in banned_sessions or session in seen_sessions or meta.get("source") != "recording":
            raise ValueError("Test session is historical, already inspected, or a correction.")
        if not meta.get("created_utc") or datetime.fromisoformat(meta["created_utc"]) <= datetime.fromisoformat(frozen["created_utc"]):
            raise ValueError("Final test recordings must be collected after the study freeze.")
        candidates.append(path)
        candidate_sessions.add(session)
        label = meta.get("label")
        if label not in LABELS:
            raise ValueError("Test recording has an unknown label.")
        test_counts[label] += 1
        session_labels.setdefault(session, set()).add(label)
    if not candidates:
        raise ValueError("No fresh post-freeze test recordings.")
    if any(test_counts[label] < 5 for label in LABELS) or any(labels != set(LABELS) for labels in session_labels.values()):
        raise ValueError("Finish test collection: at least five/class overall and all eight labels in each session.")
    sessions = sorted(candidate_sessions)
    # Mark access before labels/coordinates are loaded: failed attempts also retire a pool.
    ledger["accesses"].append({"created_utc": utc_now(), "study": str(run.resolve()),
                              "sessions": sessions, "files": [str(p.resolve()) for p in candidates], "state": "accessed"})
    write_json(ledger_path, ledger)
    samples, records, identities = [], [], set()
    for path in candidates:
        digest = sha256(path)
        xyz, times, visible, meta = load_clip(path)
        identity = content_sha(xyz, times, visible, meta["aspect"])
        if identity in banned_contents or identity in identities:
            raise ValueError("Test raw contents duplicate development or another test clip.")
        identities.add(identity)
        samples.append((features(xyz, times, visible, meta["aspect"]), LABELS.index(meta["label"]), str(path.resolve())))
        records.append({"file": str(path.resolve()), "sha256": digest, "session": meta["session"], "label": meta["label"]})
    ledger["accesses"][-1]["raw_content_sha256"] = sorted(identities)
    write_json(ledger_path, ledger)
    require_classes(samples, "test")
    for session in sessions:
        if {r["label"] for r in records if r["session"] == session} != set(LABELS):
            raise ValueError("Each final test session must cover all eight labels.")
    output = run / "final_test"
    output.mkdir(exist_ok=False)
    write_json(output / "manifest.json", {"records": records, "freeze_sha256": sha256(run / "freeze.json")})
    x, y = tensors(samples)
    np.savez_compressed(output / "features.npz", x=x.numpy(), y=y.numpy())
    results = []
    torch.set_num_threads(2)
    for r in frozen["checkpoints"]:
        model, checkpoint = load_model(run / r["file"])
        if set(sessions).intersection(checkpoint["development_sessions"]):
            raise ValueError("Test sessions overlap checkpoint development.")
        with torch.inference_mode():
            p = model(x).softmax(1).numpy()
        result = metrics(p, y.numpy())
        result["seed"] = r["seed"]
        result["sessions"] = {}
        for session in sessions:
            indices = [i for i, item in enumerate(records) if item["session"] == session]
            result["sessions"][session] = metrics(p[indices], y.numpy()[indices])
        write_json(output / f"seed_{r['seed']}_predictions.json", predictions(samples, p))
        results.append(result)
    for r in records:
        if sha256(r["file"]) != r["sha256"]:
            raise ValueError("Test input changed during evaluation.")
    write_json(output / "summary.json", summarize(results))
    ledger["accesses"][-1]["state"] = "complete"
    ledger["accesses"][-1]["report"] = str(output.resolve())
    write_json(ledger_path, ledger)
    print(f"Final test saved: {output}. These sessions are now inspected; preserve results without tuning on them.")
    return output


def record(folder, argv, hand="unspecified", notes=""):
    require_contract(folder)
    from gesture_commands.app import main as camera_main
    profile = SimpleNamespace(data=Path(folder), keys=KEYS, labels=LABELS, save_clip=save_clip,
                              gate=CommandGate, load_model=load_model,
                              window="Eight Gesture Commands v2 - Screen Demo",
                              metadata={"taxonomy": TAXONOMY, "command_contract": contract(),
                                        "hand_reported": hand, "collection_notes": notes,
                                        "acquisition": "fresh_recording"})
    return camera_main(profile=profile, argv=argv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare")
    sub.add_parser("status")
    r = sub.add_parser("record")
    r.add_argument("--split", choices=("train", "validation", "test"), default="train")
    r.add_argument("--model", type=Path)
    r.add_argument("--camera", type=int, default=0)
    r.add_argument("--hand", choices=("left", "right", "unspecified"), default="unspecified")
    r.add_argument("--notes", default="")
    t = sub.add_parser("train")
    t.add_argument("--output", type=Path, default=OUTPUT)
    for name in ("freeze", "test"):
        p = sub.add_parser(name)
        p.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare(args.data)
        elif args.action == "status":
            inventory(args.data)
        elif args.action == "train":
            train(args.data, args.output)
        elif args.action == "record":
            argv = ["--split", args.split, "--camera", str(args.camera)]
            if args.model:
                argv += ["--model", str(args.model)]
            return record(args.data, argv, args.hand, args.notes)
        elif args.action == "freeze":
            freeze(args.run)
        elif args.action == "test":
            evaluate_test(args.run, args.data)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
