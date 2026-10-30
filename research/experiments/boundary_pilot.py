"""Guided, isolated collection for a class-matched gesture-data comparison.

Reuses the existing camera/replay and eight-class recording format. This module
prepares, records and audits data; fitting follows a separate dataset audit.
It never loads a gesture checkpoint, trains, or opens test recordings.
"""

import argparse
import json
import textwrap
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from gesture_commands.data import quality
from . import commands_v2 as v2
from .evaluate_v2_sessions import DEFAULT_STUDY, check_study
from .run import PROJECT, sha256, write_json

EXPERIMENT = "004_boundary_data"
ROOT = PROJECT / "research/experiments/data/boundary_pilot"
EVALUATION = DEFAULT_STUDY / "session_evaluations/20261008T052213777935Z"
FOCUS = ("FORWARD", "BACKWARD", "NO_COMMAND", "DOWN")
GROUPS = ("routine", "targeted")
KEY = {label: key for key, label in v2.KEYS.items()}

ROUTINE = {
    "FORWARD": "Use your usual comfortable, clear come-toward-you gesture.",
    "BACKWARD": "Use your usual comfortable, clear push-away gesture.",
    "NO_COMMAND": "Rest or move naturally as usual; give no deliberate command.",
    "DOWN": "Use your usual comfortable, deliberate DOWN gesture.",
}
TARGETED = {
    "FORWARD": ["Clearly beckon toward you with one finger.",
                "Clearly beckon toward you with several fingers.",
                "Clearly beckon toward you with an open hand.",
                "Beckon toward you at a slightly different hand angle.",
                "Beckon toward you at a different comfortable distance."],
    "BACKWARD": ["Clearly push your hand away from you.",
                 "Clearly push away with several fingers extended.",
                 "Clearly push away with an open hand.",
                 "Push away at a slightly different hand angle.",
                 "Push away at a different comfortable distance."],
    "NO_COMMAND": ["Keep your naturally lowered hand resting and relaxed.",
                   "Rest with relaxed fingers downward; do not point on purpose.",
                   "Lower/reposition your relaxed hand normally, without a command.",
                   "Hold a relaxed resting hand at a slightly different angle.",
                   "Move/reposition normally; avoid any deliberate command."],
    "DOWN": ["Deliberately point one finger clearly downward.",
             "Deliberately point several fingers clearly downward.",
             "Clearly signal DOWN with deliberate whole-hand downward motion.",
             "Deliberately point down at a slightly different hand angle.",
             "Clearly signal DOWN at a different comfortable distance."],
}


def schedule(sitting):
    if type(sitting) is not int or sitting not in (1, 2):
        raise ValueError("Use sitting 1 or 2.")
    order = GROUPS if sitting == 1 else tuple(reversed(GROUPS))
    return [{"index": i, "group": group, "label": label, "repeat": repeat,
             "hint": ROUTINE[label] if group == "routine" else TARGETED[label][repeat]}
            for i, (group, label, repeat) in enumerate(
                (g, label, r) for g in order for label in FOCUS for r in range(5))]


def now():
    return datetime.now(timezone.utc).isoformat()


def read_plan(root):
    root = Path(root)
    path = root / "plan.json"
    if sha256(path) != (root / "plan.sha256").read_text().strip():
        raise ValueError("Collection plan changed; preserve it and version a new plan.")
    plan = json.loads(path.read_text())
    if (plan["experiment_id"] != EXPERIMENT or plan["contract"] != v2.contract()
            or plan["schedules"] != {str(s): schedule(s) for s in (1, 2)}):
        raise ValueError("Collection code/contract and saved plan differ.")
    if sha256(Path(__file__)) != plan["collector_sha256"]:
        raise ValueError("Collector changed after preparation; version a new plan.")
    _, candidates, _ = check_study(Path(plan["base_study"]))
    if candidates != plan["base_checkpoints"]:
        raise ValueError("Reference candidates differ from the collection plan.")
    for record in plan["base_records"]:
        if sha256(record["file"]) != record["sha256"]:
            raise ValueError(f"Frozen development input changed: {record['file']}")
    return plan


def prepare(root=ROOT, study=DEFAULT_STUDY, evaluation=EVALUATION):
    root, study, evaluation = map(Path, (root, study, evaluation))
    if (root / "plan.json").exists():
        return read_plan(root)
    env, candidates, _ = check_study(study)
    original = json.loads((study / "manifest.json").read_text())
    new = json.loads((evaluation / "manifest.json").read_text())
    if original["contract"] != v2.contract() or new["contract"] != v2.contract():
        raise ValueError("Reference studies use different gesture contracts.")
    if json.loads((evaluation / "status.json").read_text()).get("state") != "complete":
        raise ValueError("Reference new-session evaluation is incomplete.")
    eval_env = json.loads((evaluation / "environment.json").read_text())
    for name, digest in eval_env["artifact_sha256"].items():
        if sha256(evaluation / name) != digest:
            raise ValueError("Reference evaluation input changed.")
    if any(r["split"] != "validation" for r in new["records"]):
        raise ValueError("Monitoring session contains a non-validation recording.")
    records = original["records"] + new["records"]
    if (len({r["file"] for r in records}) != len(records)
            or len({r["raw_content_sha256"] for r in records}) != len(records)):
        raise ValueError("Reference recordings overlap.")
    if {r["session"] for r in original["records"]} & {r["session"] for r in new["records"]}:
        raise ValueError("Reference sessions overlap.")
    for record in records:
        if sha256(record["file"]) != record["sha256"]:
            raise ValueError("A reference recording changed.")
    plan = {"experiment_id": EXPERIMENT, "created_utc": now(),
            "contract": v2.contract(), "base_study": str(study.resolve()),
            "monitor_evaluation": str(evaluation.resolve()), "base_records": records,
            "base_checkpoints": candidates, "base_source_sha256": env["source_sha256"],
            "collector_sha256": sha256(Path(__file__)),
            "schedules": {str(s): schedule(s) for s in (1, 2)},
            "planned_additions_per_model": 40, "new_human_labels_total": 80,
            "training_config": v2.TRAIN_CONFIG,
            "selection_validation_files": [r["file"] for r in original["records"] if r["split"] == "validation"],
            "monitor_validation_files": [r["file"] for r in new["records"]],
            "primary_comparison": "targeted minus routine mean macro F1 on the inspected monitoring session",
            "primary_scope": "single-user supervised development pilot, no novelty or label-savings claim",
            "fitting_status": "pending collection and dataset audit; no fitting in this module"}
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "plan.json", plan)
    (root / "plan.sha256").write_text(sha256(root / "plan.json") + "\n")
    (root / "recordings").mkdir(exist_ok=True)
    return plan


def audit(root=ROOT):
    root = Path(root)
    plan = read_plan(root)
    base_sessions = {r["session"] for r in plan["base_records"]}
    seen_raw = {r["raw_content_sha256"] for r in plan["base_records"]}
    slots, session_sittings, records = {}, {}, []
    for path in sorted((root / "recordings").glob("*.npz")):
        digest = sha256(path)
        xyz, times, visible, meta = v2.load_clip(path)
        sitting, index = meta.get("collection_sitting"), meta.get("task_index")
        if (type(sitting) is not int or sitting not in (1, 2) or type(index) is not int
                or not 0 <= index < 40):
            raise ValueError("Recording lacks a valid sitting/task identity.")
        task = plan["schedules"][str(sitting)][index]
        if (meta.get("experiment_id") != EXPERIMENT or meta["split"] != "train"
                or meta.get("source") != "recording" or meta.get("model") is not None
                or meta.get("prediction") is not None or meta.get("scores") is not None
                or meta.get("collection_group") != task["group"] or meta["label"] != task["label"]
                or meta.get("task_hint") != task["hint"]
                or meta.get("collection_plan_sha256") != sha256(root / "plan.json")):
            raise ValueError("Recording does not match its frozen human-label task.")
        if meta["session"] in base_sessions:
            raise ValueError("New training recording reuses a development session.")
        if meta["session"] in session_sittings and session_sittings[meta["session"]] != sitting:
            raise ValueError("One capture session crosses collection sittings.")
        session_sittings[meta["session"]] = sitting
        slot = (sitting, index)
        if slot in slots:
            raise ValueError("Two recordings occupy the same task slot.")
        identity = v2.content_sha(xyz, times, visible, meta["aspect"])
        if identity in seen_raw:
            raise ValueError("Duplicate raw contents in new or reference recordings.")
        seen_raw.add(identity)
        if sha256(path) != digest:
            raise ValueError("Recording changed during audit.")
        record = {"file": str(path.resolve()), "sha256": digest,
                  "raw_content_sha256": identity, "session": meta["session"],
                  "sitting": sitting, "task_index": index, "group": task["group"],
                  "label": meta["label"], "passes_tracking_quality": quality(times, visible)}
        slots[slot] = record
        records.append(record)
    counts = {s: sum(r["sitting"] == s for r in records) for s in (1, 2)}
    for sitting, count in counts.items():
        if {index for (s, index) in slots if s == sitting} != set(range(count)):
            raise ValueError("Task slots are not a completed prefix; do not skip tasks.")
    if counts[2] and counts[1] != 40:
        raise ValueError("Complete sitting 1 before beginning sitting 2.")
    return {"records": records, "sittings": {str(s): {"saved": counts[s], "required": 40} for s in counts},
            "groups": {g: dict(Counter(r["label"] for r in records if r["group"] == g)) for g in GROUPS},
            "tracking_quality_failures": [r["file"] for r in records if not r["passes_tracking_quality"]],
            "collection_complete": len(records) == 80,
            "next_step": "dataset audit and matched fitting after collection; no automatic training"}


class PromptKeys(dict):
    """Keep camera's existing label interface; accept only the prompted label."""

    def __init__(self, current_task):
        super().__init__((KEY[label], label) for label in FOCUS)
        self.current_task = current_task

    def __iter__(self):
        task = self.current_task()
        return iter(()) if task is None else iter((KEY[task["label"]],))


def guided_lines(task, saved, original):
    if task is None:
        return ["Sitting complete: 40/40 saved. Press q to finish.",
                "No model was trained; recordings are preserved."]
    header = f"Clip {saved+1}/40 | {task['group'].upper()} | {task['label']} | save: {KEY[task['label']]}"
    return [original[0], header, *textwrap.wrap(task["hint"], width=65),
            original[1], "SPACE review | prompted key saves | x repeat | q pause"]


def record(sitting, root=ROOT, camera=0, hand="unspecified"):
    root = Path(root)
    state = audit(root)
    if sitting not in (1, 2):
        raise ValueError("Use sitting 1 or 2.")
    if sitting == 2 and state["sittings"]["1"]["saved"] != 40:
        raise ValueError("Complete sitting 1 before beginning sitting 2.")
    plan = read_plan(root)
    tasks = plan["schedules"][str(sitting)]
    saved = state["sittings"][str(sitting)]["saved"]
    if saved == len(tasks):
        print(f"Sitting {sitting} already complete; no camera opened.")
        return 0
    attempt = uuid4().hex
    started = now()
    start_clock = time.monotonic()
    saved_at_start = saved

    def current_task():
        return tasks[saved] if saved < len(tasks) else None

    def save_prompted(folder, xyz, times, visible, metadata):
        nonlocal saved
        task = current_task()
        if task is None or metadata["label"] != task["label"]:
            raise ValueError("Label differs from the prompted task; discard and repeat.")
        if (metadata["split"] != "train" or metadata.get("source") != "recording"
                or metadata.get("session") in {r["session"] for r in plan["base_records"]}):
            raise ValueError("Recorder must collect fresh human-labeled training data.")
        path = v2.save_clip(folder, xyz, times, visible,
                            {**metadata, "experiment_id": EXPERIMENT, "collection_sitting": sitting,
                             "collection_group": task["group"], "task_index": saved,
                             "task_hint": task["hint"], "recording_attempt": attempt,
                             "collection_plan_sha256": sha256(root / "plan.json")})
        saved += 1
        next_task = current_task()
        print("Sitting complete; press q." if next_task is None else
              f"Next {saved+1}/40: {next_task['group']} {next_task['label']} — {next_task['hint']}", flush=True)
        return path

    from gesture_commands import app as camera_app
    original_overlay = camera_app.overlay

    def prompted_overlay(cv2, frame, lines):
        return original_overlay(cv2, frame, guided_lines(current_task(), saved, lines))

    profile = SimpleNamespace(data=root / "recordings", keys=PromptKeys(current_task), labels=v2.LABELS,
                              save_clip=save_prompted, gate=v2.CommandGate, load_model=v2.load_model,
                              window="Gesture data comparison - guided recording",
                              metadata={"hand_reported": hand, "acquisition": "planned_supervised_collection",
                                        "command_contract": v2.contract()})
    first = current_task()
    print(f"Sitting {sitting}, next {saved+1}/40: {first['group']} {first['label']} — {first['hint']}")
    print("Keep the same camera/hand within this sitting. Read prompts and review each replay.")
    print("Only the prompted label key saves. x repeats; q pauses. Resume only the same sitting.")
    camera_app.overlay = prompted_overlay  # Scoped adapter; no existing source file is edited.
    outcome = "paused"
    try:
        result = camera_app.main(profile=profile, argv=["--split", "train", "--camera", str(camera)])
        outcome = "complete" if saved == len(tasks) else "paused"
        return result
    except BaseException:
        outcome = "failed"
        raise
    finally:
        camera_app.overlay = original_overlay
        attempts = root / "recording_attempts"
        attempts.mkdir(exist_ok=True)
        write_json(attempts / f"{attempt}.json", {"started_utc": started, "ended_utc": now(),
                   "wall_seconds": time.monotonic()-start_clock, "sitting": sitting,
                   "saved_before": saved_at_start, "saved_after": saved, "state": outcome,
                   "scope": "camera/replay/preparation/pause interval; not pure annotation time"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare")
    sub.add_parser("status")
    r = sub.add_parser("record")
    r.add_argument("--sitting", type=int, choices=(1, 2), required=True)
    r.add_argument("--camera", type=int, default=0)
    r.add_argument("--hand", choices=("left", "right", "unspecified"), default="unspecified")
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            plan = prepare(args.data)
            print(f"Prepared {EXPERIMENT}: 80 new labels, 40 added per compared model.")
            print(f"Sittings: {list(plan['schedules'])}. Data: {args.data}")
        elif args.action == "status":
            print(json.dumps(audit(args.data), indent=2))
        else:
            return record(args.sitting, args.data, args.camera, args.hand)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
