"""Lock the complete matched study, then collect and jointly evaluate fresh clips.

No fitting, model-assisted labeling, ensemble, app activation or threshold tuning.
Collection/status read metadata only. Payload access retires the entire test pool,
including failed evaluations. This is a single-user clip pilot, not a robot test.
"""

import argparse
import json
import shutil
import textwrap
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import torch

from . import boundary_train as training
from . import commands_v2 as v2
from .run import PROJECT, sha256, write_json

ROOT = PROJECT / "research/experiments/data/boundary_test"
STUDY = training.OUTPUT / "20261008T072040903427Z"
EXPERIMENT = "005_boundary_frozen_test"
KEY = {label: key for key, label in v2.KEYS.items()}
SOURCES = tuple(dict.fromkeys((*training.SOURCES,
    "research/experiments/boundary_test.py", "research/experiments/test_boundary_test.py",
    "research/BOUNDARY_TEST_PLAN.md")))
STYLES = {
    "FORWARD": ["one finger", "several fingers", "open hand", "a modestly different angle", "a comfortable usual style"],
    "BACKWARD": ["a comfortable push-away", "several fingers", "open hand", "a modestly different angle", "a comfortable usual style"],
    "LEFT": ["one finger", "several fingers", "a comfortable usual style", "a modestly different angle", "a comfortable usual style"],
    "RIGHT": ["one finger", "several fingers", "a comfortable usual style", "a modestly different angle", "a comfortable usual style"],
    "STOP": ["a comfortable usual fist", "a slightly different hand height", "a comfortable usual fist", "a modestly different angle", "a comfortable usual fist"],
    "NO_COMMAND": ["visible relaxed resting hand", "naturally lowered relaxed hand at rest", "ordinary repositioning without a command", "resting fingers facing downward without a command", "no hand in view"],
    "UP": ["one finger pointing up", "several fingers pointing up", "deliberate whole-hand upward movement", "pointing up at a modestly different angle", "deliberate whole-hand upward movement"],
    "DOWN": ["one finger pointing down", "several fingers pointing down", "deliberate whole-hand downward movement", "pointing down at a modestly different angle", "deliberate whole-hand downward movement"],
}


def schedule(sitting):
    if sitting not in (1, 2):
        raise ValueError("Use sitting 1 or 2.")
    labels = v2.LABELS if sitting == 1 else tuple(reversed(v2.LABELS))
    return [{"index": j*8+i, "label": label,
             "hint": f"{v2.MEANINGS[label]} This clip: {STYLES[label][j]}."}
            for j in range(5) for i, label in enumerate(labels)]


def freeze(root=ROOT, study=STUDY, migration=v2.DATA / "migration.json"):
    root, study, migration = map(Path, (root, study, migration))
    if root.exists() and any(root.iterdir()):
        raise ValueError("Freeze destination already contains artifacts; preserve them.")
    config = training.read_json(study / "config.json")
    seeds = config["training"]["seeds"]
    if config["conditions"] != list(training.CONDITIONS) or training.read_json(study/"status.json") != {
            "state": "complete", "trained_trials": 2*len(seeds), "evaluated_candidates": 3*len(seeds)}:
        raise ValueError("Require the complete reference/routine/targeted study.")
    torch.set_num_threads(config["training"]["threads"])
    verified = training.verify(study)
    expected = {f"{c}/seed_{seed}/model.pt" for c in training.CONDITIONS for seed in seeds}
    if set(verified["checkpoints_sha256"]) != expected:
        raise ValueError("Candidate membership differs from declared conditions/seeds.")
    env = training.read_json(study/"environment.json")
    # Document updates are allowed; code that produced the study must still match.
    for name, digest in env["source_sha256"].items():
        if name.endswith(".py") and sha256(PROJECT/name) != digest:
            raise ValueError("Study code changed before freeze.")
    manifest = training.read_json(study/"manifest.json")
    historical = training.read_json(migration)
    plan = {"experiment_id": EXPERIMENT, "created_utc": v2.utc_now(),
            "study": str(study.resolve()), "contract": v2.contract(), "participant": "current_user",
            "checkpoints": [{"condition": c, "seed": seed, "file": f"{c}/seed_{seed}/model.pt",
                             "sha256": verified["checkpoints_sha256"][f"{c}/seed_{seed}/model.pt"]}
                            for c in training.CONDITIONS for seed in seeds],
            "study_sha256": {name: sha256(study/name) for name in
                ("config.json", "manifest.json", "development_features.npz", "trials.json", "summary.json", "environment.json", "status.json")},
            "migration": {"file": str(migration.resolve()), "sha256": sha256(migration)},
            "banned_sessions": sorted(set(historical["historical_sessions"]) | {r["session"] for r in manifest["records"]}),
            "banned_raw_contents": sorted(set(historical["historical_development_raw_hashes"]) | {r["raw_content_sha256"] for r in manifest["records"]}),
            "source_sha256": {name: sha256(PROJECT/name) for name in SOURCES},
            "schedules": {str(s): schedule(s) for s in (1,2)}, "required_sittings": 2, "clips_per_sitting": 40,
            "primary": "Pooled 80-clip macro F1: targeted minus routine, paired by all declared seeds; average seed differences.",
            "supporting": list(training.FIELDS) + ["per-class precision/recall/F1", "confusion matrices", "each sitting separately", "tracking quality", "acquisition time"],
            "rule": "All candidates, raw eight-class scores; no ensemble, winner-seed selection, tuning, retraining or early per-sitting scoring.",
            "interpretation": "Two fresh sittings of the same user; shared clips and seed spread do not establish unseen-person or robot reliability, label savings, novelty or stage-order advantage."}
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/"contract.json", v2.contract())
    for name in SOURCES:
        destination = root/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/name, destination)
    write_json(root/"preparation_verification.json", verified)
    write_json(root/"freeze.json", plan)
    (root/"freeze.sha256").write_text(sha256(root/"freeze.json")+"\n")
    print(f"Frozen {len(expected)} candidates. Collect two new 40-clip sittings; no test scored.")
    return plan


def read_freeze(root=ROOT):
    root = Path(root)
    if sha256(root/"freeze.json") != (root/"freeze.sha256").read_text().strip():
        raise ValueError("Frozen policy changed.")
    plan = training.read_json(root/"freeze.json")
    if plan["contract"] != v2.contract() or plan["schedules"] != {str(s): schedule(s) for s in (1,2)}:
        raise ValueError("Frozen command contract or schedule differs.")
    v2.require_contract(root)
    for name, digest in plan["source_sha256"].items():
        if sha256(root/"source"/name) != digest or (name.endswith(".py") and sha256(PROJECT/name) != digest):
            raise ValueError("Frozen source changed.")
    study = Path(plan["study"])
    for name, digest in plan["study_sha256"].items():
        if sha256(study/name) != digest:
            raise ValueError("Frozen development artifact changed.")
    for r in training.read_json(study/"manifest.json")["records"]:
        if sha256(r["file"]) != r["sha256"]:
            raise ValueError("Development recording changed after freeze.")
    for candidate in plan["checkpoints"]:
        if sha256(study/candidate["file"]) != candidate["sha256"]:
            raise ValueError("Frozen checkpoint changed.")
    if sha256(plan["migration"]["file"]) != plan["migration"]["sha256"]:
        raise ValueError("Historical provenance changed.")
    return plan


def audit(root=ROOT):
    """Metadata-only progress/provenance check; no features or model predictions."""
    root = Path(root)
    plan = read_freeze(root)
    rows, sessions, slots = [], {}, {1: {}, 2: {}}
    for path in sorted((root/"recordings").glob("*.npz")):
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(str(a["metadata"]))
        sitting, index, session = meta.get("collection_sitting"), meta.get("task_index"), meta.get("session")
        if sitting not in slots or not isinstance(index, int) or index not in range(40):
            raise ValueError("Foreign or unplanned test metadata; no payload accessed.")
        task = plan["schedules"][str(sitting)][index]
        expected = {"split": "test", "source": "recording", "label": task["label"], "task_hint": task["hint"],
                    "schema_version": v2.VERSION, "taxonomy": v2.TAXONOMY, "feature_version": 1,
                    "convention": v2.CONVENTION, "labels": list(v2.LABELS), "experiment_id": EXPERIMENT,
                    "participant": plan["participant"], "freeze_sha256": sha256(root/"freeze.json")}
        if any(meta.get(k) != value for k, value in expected.items()) or any(meta.get(k) is not None for k in ("model", "prediction", "scores")):
            raise ValueError("Test metadata violates frozen human-only policy; no payload accessed.")
        if not isinstance(session, str) or not session or session in plan["banned_sessions"]:
            raise ValueError("Historical or invalid test session; no payload accessed.")
        created = datetime.fromisoformat(meta.get("created_utc", ""))
        if created.tzinfo is None or created <= datetime.fromisoformat(plan["created_utc"]) or created > datetime.now(timezone.utc):
            raise ValueError("Test recording must be made after freeze, not future-dated.")
        if session in sessions and sessions[session] != sitting:
            raise ValueError("Capture session crosses physical sittings.")
        if index in slots[sitting]:
            raise ValueError("Repeated test task; preserve and resolve provenance before scoring.")
        sessions[session] = sitting
        row = {"file": str(path.resolve()), "sha256": sha256(path), "session": session,
               "sitting": sitting, "task_index": index, "label": meta["label"], "created_utc": meta["created_utc"]}
        slots[sitting][index] = row
        rows.append(row)
    for sitting, assigned in slots.items():
        if set(assigned) != set(range(len(assigned))):
            raise ValueError("Test tasks must form a saved prefix, with no missing slots.")
    if slots[2] and len(slots[1]) != 40:
        raise ValueError("Finish sitting 1 before sitting 2.")
    ledger = training.read_json(root/"test_access.json") if (root/"test_access.json").exists() else {"accesses": []}
    retired = {s for a in ledger["accesses"] for s in a["sessions"]}
    return {"sittings": {str(s): {"saved": len(slots[s]), "required": 40,
                                  "labels": dict(Counter(r["label"] for r in slots[s].values()))} for s in (1,2)},
            "complete": all(len(slots[s]) == 40 for s in (1,2)), "inspected": bool(retired.intersection(sessions)),
            "records": sorted(rows, key=lambda r: (r["sitting"], r["task_index"]))}


class PromptKeys(dict):
    def __init__(self, current):
        super().__init__(v2.KEYS)
        self.current = current

    def __iter__(self):
        task = self.current()
        return iter(()) if task is None else iter((KEY[task["label"]],))


def record(sitting, root=ROOT, camera=0, hand="unspecified"):
    root = Path(root)
    tasks = schedule(sitting)
    state, plan = audit(root), read_freeze(root)
    if state["inspected"] or (root/"test_access.json").exists():
        raise ValueError("Test pool has been accessed; preserve it without extending collection.")
    if sitting == 2 and state["sittings"]["1"]["saved"] != 40:
        raise ValueError("Complete sitting 1 first, then begin a genuinely new sitting.")
    saved = initial = state["sittings"][str(sitting)]["saved"]
    if saved == 40:
        print("Sitting complete; no camera opened.")
        return 0
    attempt, started, clock = uuid4().hex, v2.utc_now(), time.monotonic()

    def current():
        return tasks[saved] if saved < 40 else None

    def save(folder, xyz, times, visible, meta):
        nonlocal saved
        task = current()
        if task is None or meta.get("label") != task["label"] or meta.get("split") != "test" or meta.get("source") != "recording" or any(meta.get(k) is not None for k in ("model", "prediction", "scores")):
            raise ValueError("Only the current human-labeled test prompt may be saved.")
        path = v2.save_clip(folder, xyz, times, visible, {**meta, "experiment_id": EXPERIMENT,
            "participant": plan["participant"], "collection_sitting": sitting, "task_index": saved,
            "task_hint": task["hint"], "freeze_sha256": sha256(root/"freeze.json"), "recording_attempt": attempt})
        saved += 1
        print("40/40 complete. Press q." if current() is None else
              f"Next {saved+1}/40: {current()['label']} — {current()['hint']}", flush=True)
        return path

    from gesture_commands import app
    original = app.overlay

    def overlay(cv2, frame, lines):
        task = current()
        message = ["40/40 saved. Press q to finish.", "No predictions or training."] if task is None else [
            lines[0], f"TEST {saved+1}/40 | {task['label']} | save: {KEY[task['label']]}",
            *textwrap.wrap(task["hint"], 65), "SPACE replay | prompted key saves | x repeat | q pause"]
        return original(cv2, frame, message)

    profile = SimpleNamespace(data=root/"recordings", keys=PromptKeys(current), labels=v2.LABELS,
        save_clip=save, gate=v2.CommandGate, load_model=v2.load_model, window="Fresh gesture test - human prompts only",
        metadata={"hand_reported": hand, "acquisition": "post_freeze_test", "command_contract": v2.contract()})
    print(f"Sitting {sitting}: next {saved+1}/40. {current()['hint']}")
    print("Perform a new 2-second clip, SPACE to replay, prompted key to save; x repeats, q pauses.")
    app.overlay, outcome = overlay, "failed"
    try:
        result = app.main(profile=profile, argv=["--split", "test", "--camera", str(camera)])
        outcome = "complete" if saved == 40 else "paused"
        return result
    finally:
        app.overlay = original
        (root/"recording_attempts").mkdir(exist_ok=True)
        write_json(root/"recording_attempts"/f"{attempt}.json", {"started_utc": started,
            "ended_utc": v2.utc_now(), "wall_seconds": time.monotonic()-clock, "sitting": sitting,
            "saved_before": initial, "saved_after": saved, "state": outcome,
            "scope": "camera/replay/preparation/pause interval, not pure labeling time"})


def summary(trials):
    seeds = sorted({r["seed"] for r in trials})
    by_key = {(r["condition"], r["seed"]): r for r in trials}
    if len(by_key) != len(trials) or set(by_key) != {(c,s) for c in training.CONDITIONS for s in seeds}:
        raise ValueError("Require every candidate exactly once.")
    aggregate = {}
    for c in training.CONDITIONS:
        aggregate[c] = {field: {"mean": float(np.mean([by_key[c,s]["pooled"][field] for s in seeds])),
                               "run_sd": float(np.std([by_key[c,s]["pooled"][field] for s in seeds], ddof=1)) if len(seeds)>1 else None}
                        for field in training.FIELDS}
    changes = {str(s): by_key["targeted",s]["pooled"]["macro_f1"]-by_key["routine",s]["pooled"]["macro_f1"] for s in seeds}
    return {"conditions": aggregate, "primary_comparison": {"field": "pooled macro F1: targeted minus routine",
        "paired_by_seed": changes, "mean": float(np.mean(list(changes.values())))},
        "interpretation": "Fresh same-user clip pilot, all seeds share 80 clips. Seed SD is optimization variation, not person/session uncertainty. No robot, label-saving, SSL or novelty claim."}


def evaluate(root=ROOT):
    root = Path(root)
    plan, state = read_freeze(root), audit(root)
    if not state["complete"]:
        raise ValueError("Finish both sittings: all 80 test clips required before payload access.")
    ledger_path = root/"test_access.json"
    if ledger_path.exists():
        raise ValueError("This test pool has already been accessed; preserve results, no repeat evaluation.")
    # Exclusive ledger creation precedes any landmark access, including failures.
    access = {"created_utc": v2.utc_now(), "state": "accessed", "sessions": sorted({r["session"] for r in state["records"]}),
              "freeze_sha256": sha256(root/"freeze.json"), "raw_content_sha256": []}
    ledger = {"accesses": [access]}
    with ledger_path.open("x") as stream:
        json.dump(ledger, stream, indent=2)
    output = root/"evaluation"
    try:
        output.mkdir(exist_ok=False)
        rows, samples, seen = state["records"], [], set(plan["banned_raw_contents"])
        for row in rows:
            xyz, times, visible, meta = v2.load_clip(row["file"])
            identity = v2.content_sha(xyz, times, visible, meta["aspect"])
            access["raw_content_sha256"].append(identity)
            write_json(ledger_path, ledger)
            if identity in seen:
                raise ValueError("Raw test clip duplicates development or another test clip; pool is now inspected.")
            seen.add(identity)
            row["raw_content_sha256"] = identity
            row["passes_tracking_quality"] = v2.quality(times, visible)
            samples.append((v2.features(xyz, times, visible, meta["aspect"]), v2.LABELS.index(row["label"]), row["file"]))
            if sha256(row["file"]) != row["sha256"]:
                raise ValueError("Test recording changed during payload access.")
        x = torch.from_numpy(np.stack([s[0] for s in samples])); y = np.array([s[1] for s in samples])
        np.savez_compressed(output/"features.npz", x=x.numpy(), y=y)
        write_json(output/"manifest.json", {"records": rows, "freeze_sha256": sha256(root/"freeze.json"),
                    "quality_failures_retained": sum(not r["passes_tracking_quality"] for r in rows)})
        torch.set_num_threads(2)
        trials = []
        for candidate in plan["checkpoints"]:
            model, checkpoint = v2.load_model(Path(plan["study"])/candidate["file"])
            if set(access["sessions"]) & set(checkpoint["development_sessions"]):
                raise ValueError("Test session overlaps candidate development.")
            with torch.inference_mode():
                p = model(x).softmax(1).numpy()
            result = {"condition": candidate["condition"], "seed": candidate["seed"],
                      "checkpoint_sha256": candidate["sha256"], "pooled": training.metric_set(p,y), "sittings": {}}
            for sitting in (1,2):
                indices = [i for i,r in enumerate(rows) if r["sitting"] == sitting]
                result["sittings"][str(sitting)] = training.metric_set(p[indices],y[indices])
            destination = output/candidate["condition"]/f"seed_{candidate['seed']}"
            destination.mkdir(parents=True)
            write_json(destination/"predictions.json", v2.predictions(samples,p))
            write_json(destination/"metrics.json", result)
            trials.append(result)
        # Detect changes to any frozen study/source/input during scoring.
        read_freeze(root)
        for row in rows:
            if sha256(row["file"]) != row["sha256"]:
                raise ValueError("Test recording changed during evaluation.")
        write_json(output/"trials.json", trials)
        write_json(output/"summary.json", summary(trials))
        attempts = [training.read_json(p) for p in sorted((root/"recording_attempts").glob("*.json"))]
        write_json(output/"costs.json", {"evaluation_labels": 80, "recording_attempts": attempts,
            "logged_camera_seconds": sum(a["wall_seconds"] for a in attempts), "training_this_phase": "none"})
        write_json(output/"artifacts_sha256.json", {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()})
        access.update(state="complete", report=str(output.resolve()))
        print(f"All {len(trials)} candidates evaluated together: {output}. This pool is now inspected.")
        return output
    except BaseException as error:
        access.update(state="failed_inspected", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(ledger_path, ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("freeze", "status", "record", "evaluate"):
        p = sub.add_parser(action)
        p.add_argument("--root", type=Path, default=ROOT)
        if action == "freeze":
            p.add_argument("--study", type=Path, default=STUDY)
        if action == "record":
            p.add_argument("--sitting", type=int, choices=(1,2), required=True)
            p.add_argument("--camera", type=int, default=0)
            p.add_argument("--hand", default="unspecified")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.root, args.study)
    elif args.action == "record":
        record(args.sitting,args.root,args.camera,args.hand)
    elif args.action == "evaluate":
        evaluate(args.root)
    else:
        state = audit(args.root)
        print(json.dumps({k:v for k,v in state.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
