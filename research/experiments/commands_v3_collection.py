"""Replacement-UP supervised collection with an explicit open-palm STOP contract.

Old experiments stay immutable. Only an explicit routine-training manifest is
eligible for reuse; old UP and current validation/test cohorts are excluded.
Earlier authorized recycled training retains its historical provenance.
This module collects data, not weights. No automatic camera-model activation.
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

import numpy as np
import torch

from . import boundary_test as previous_test
from . import boundary_train as previous_training
from . import commands_v2 as v2
from .run import PROJECT, sha256, write_json

ROOT = PROJECT / "research/experiments/data/commands_v3"
STUDY = previous_test.STUDY
EXPERIMENT = "006_gesture_contract_repair"
TAXONOMY = "commands_v3"
VERSION = 3
UP_STYLES = {
    "index_up": "Point only the index finger upward; fold the other fingers and thumb. Hold the deliberate pose for about two seconds. No open-palm UP.",
    "two_fingers_up": "Point index and middle fingers upward; fold the other fingers and thumb. Hold the deliberate pose for about two seconds. No open-palm UP.",
    "thumbs_up": "Point the thumb upward; fold all four fingers. Hold the deliberate pose for about two seconds. No open-palm UP.",
    "three_shapes": "Signal UP using one index finger, index and middle fingers, or the thumb pointing upward, with the remaining fingers folded. Hold the deliberate pose for about two seconds. No open-palm UP.",
}
SHAPE_OPTIONS = {"three_shapes": ("index_up", "two_fingers_up", "thumbs_up")}
VARIATIONS = ("Use a comfortable central position.", "Use a slightly different hand height.",
              "Use a modest angle, keeping the indicated finger clearly upward.",
              "Use a slightly nearer comfortable distance, whole hand visible.",
              "Use a slightly farther comfortable distance, whole hand visible.")
SOURCES = ("research/experiments/commands_v3_collection.py", "research/experiments/commands_v2.py",
           "gesture_commands/app.py", "gesture_commands/data.py", "gesture_commands/model.py")


def contract(up_style):
    if up_style not in UP_STYLES:
        raise ValueError("Choose an explicit supported UP gesture.")
    meanings = {**v2.MEANINGS, "STOP": "Hold a still open palm facing the camera, fingers extended; no closed fist.",
                "UP": UP_STYLES[up_style]}
    return {"taxonomy": TAXONOMY, "schema_version": VERSION, "feature_version": 1,
            "labels": list(v2.LABELS), "convention": v2.CONVENTION, "meanings": meanings,
            "up_style": up_style, "up_motion_requirement": "static deliberate pose; incidental motion allowed",
            "allowed_up_shapes": list(SHAPE_OPTIONS.get(up_style,(up_style,))),
            "down_styles": ["pointing", "whole_hand_motion"], "robot_mapping": "unassigned; screen-only labels"}


def tasks(up_style,sitting=1):
    contract(up_style)
    if sitting not in (1,2):
        raise ValueError("Use sitting1 or2.")
    shapes = SHAPE_OPTIONS.get(up_style,(up_style,))
    return [{"index": i, "label": "UP", "up_shape": shapes[(i+sitting-1)%len(shapes)],
             "hint": f"{UP_STYLES[shapes[(i+sitting-1)%len(shapes)]]} {VARIATIONS[i//5]}"} for i in range(25)]


def save_clip(folder, xyz, times, visible, meta, expected_contract, filename=None):
    xyz,times,visible = v2.validate_clip(xyz,times,visible)
    if meta.get("label") not in v2.LABELS or meta.get("split") not in ("train","validation"):
        raise ValueError("Only v3 development labels/splits are supported here; no final-test imports.")
    if not isinstance(meta.get("session"),str) or not meta["session"]:
        raise ValueError("A recording session is required.")
    if not np.isfinite(meta.get("aspect",np.nan)) or meta["aspect"]<=0:
        raise ValueError("Invalid camera aspect ratio.")
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=True)
    path = folder/(filename or f"{uuid4().hex}.npz")
    metadata = {**meta,"schema_version":VERSION,"feature_version":1,"taxonomy":TAXONOMY,
                "labels":list(v2.LABELS),"convention":v2.CONVENTION,"command_contract":expected_contract}
    with path.open("xb") as stream:
        np.savez_compressed(stream,xyz=xyz,times=times-times[0],visible=visible,metadata=json.dumps(metadata))
    return path


def prepare(up_style, stop_history, root=ROOT, study=STUDY, history_ledger=previous_test.ROOT/"test_access.json",stop_history_notes=None):
    root,study = Path(root),Path(study)
    expected_contract = contract(up_style)
    if stop_history not in ("open_palm","mixed","closed_fist"):
        raise ValueError("Report the old STOP recording history explicitly.")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Destination already contains artifacts; preserve this version.")
    config = previous_training.read_json(study/"config.json")
    seeds = config["training"]["seeds"]
    if previous_training.read_json(study/"status.json") != {"state":"complete","trained_trials":2*len(seeds),"evaluated_candidates":3*len(seeds)}:
        raise ValueError("Require the completed source study.")
    torch.set_num_threads(config["training"]["threads"])
    verified = previous_training.verify(study)
    checkpoint_file = f"routine/seed_{seeds[0]}/model.pt"
    _,checkpoint = v2.load_model(study/checkpoint_file)
    manifest = previous_training.read_json(study/"manifest.json")
    by_file = {r["file"]:r for r in manifest["records"]}
    selected = [by_file[f] for f in checkpoint["training_files"]]
    # Added-cohort manifest rows carry a group instead of a split; verify the
    # actual saved split from metadata rather than guessing from row structure.
    for row in selected:
        with np.load(row["file"],allow_pickle=False) as a:
            meta=json.loads(str(a["metadata"]))
        if meta.get("split")!="train" or any(meta.get(k)!=row[k] for k in ("label","session")):
            raise ValueError("Source selection includes evaluation/mismatched data; no payload imported.")
    # Every routine seed must have exactly the same training cohort.
    for seed in seeds:
        _,other = v2.load_model(study/f"routine/seed_{seed}/model.pt")
        if other["training_files"]!=checkpoint["training_files"]:
            raise ValueError("Routine seed training cohorts differ.")
    excluded = [{**r,"reason":"replaced_up_definition" if r["label"]=="UP" else "stop_semantics_require_review"}
                for r in selected if r["label"]=="UP" or (r["label"]=="STOP" and stop_history!="open_palm")]
    included = [r for r in selected if r["label"]!="UP" and (r["label"]!="STOP" or stop_history=="open_palm")]
    historical_sessions = {r["session"] for r in manifest["records"]}
    historical_contents = {r["raw_content_sha256"] for r in manifest["records"]}
    if history_ledger is not None and Path(history_ledger).exists():
        for a in previous_training.read_json(history_ledger)["accesses"]:
            historical_sessions.update(a["sessions"])
            historical_contents.update(a.get("raw_content_sha256",[]))
    created = v2.utc_now()
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/"contract.json",expected_contract)
    write_json(root/"preparation_status.json",{"state":"copying"})
    imports = []
    try:
        for row in included:
            if sha256(row["file"])!=row["sha256"]:
                raise ValueError("Source training recording changed.")
            xyz,times,visible,meta = v2.load_clip(row["file"])
            if meta["split"]!="train" or any(meta[k]!=row[k] for k in ("label","session")):
                raise ValueError("Source manifest/metadata differ.")
            destination = save_clip(root/"reused_training",xyz,times,visible,{
                "label":meta["label"],"split":"train","session":meta["session"],"aspect":meta["aspect"],
                "source":meta.get("source","unknown"),"acquisition":"reused_authorized_routine_training",
                "origin_file":row["file"],"origin_sha256":row["sha256"],"original_metadata":meta,
                "stop_semantics_basis":(stop_history_notes or "User reports open-palm STOP; not independently reviewed.") if meta["label"]=="STOP" else None,
                "migration_utc":created},expected_contract,filename=Path(row["file"]).name)
            if sha256(row["file"])!=row["sha256"]:
                raise ValueError("Source changed while copying.")
            imports.append({"file":str(destination.resolve()),"sha256":sha256(destination),
                "origin_file":row["file"],"origin_sha256":row["sha256"],"label":row["label"],
                "session":row["session"],"raw_content_sha256":row["raw_content_sha256"]})
        plan = {"experiment_id":EXPERIMENT,"created_utc":created,"contract":expected_contract,
            "stop_history_reported":stop_history,"stop_history_notes":stop_history_notes,
            "stop_independently_reviewed":False,"source_study":str(study.resolve()),
            "source_checkpoint":checkpoint_file,"source_checkpoint_sha256":verified["checkpoints_sha256"][checkpoint_file],
            "source_manifest_sha256":sha256(study/"manifest.json"),"imports":imports,"excluded_training":excluded,
            "base_counts":dict(Counter(r["label"] for r in imports)),"planned_new_up":50,"sittings":2,
            "clips_per_sitting":25,"tasks":{str(s):tasks(up_style,s) for s in (1,2)},"banned_sessions":sorted(historical_sessions),
            "banned_raw_contents":sorted(historical_contents),"source_sha256":{s:sha256(PROJECT/s) for s in SOURCES},
            "stop_ready":stop_history=="open_palm",
            "validation_policy":"No current validation/test imported; earlier authorized recycled training retains provenance. New definition requires fresh whole-session development validation before fitting/selection; a later untouched test after choices freeze.",
            "scope":"Supervised gesture-definition revision, not an ML algorithm/label-saving result. No old weights activated or fine-tuned."}
        write_json(root/"plan.json",plan)
        (root/"plan.sha256").write_text(sha256(root/"plan.json")+"\n")
        write_json(root/"preparation_status.json",{"state":"complete","imports":len(imports)})
        print(f"Prepared v3: {len(imports)} retained training clips; old UP excluded. Collect 25 new UP clips in each of two sittings.")
        if not plan["stop_ready"]:
            print("Old STOP excluded too: review/replacement open-palm STOP examples required before training.")
        return plan
    except BaseException as error:
        write_json(root/"preparation_status.json",{"state":"failed","imports":len(imports),"error":str(error)})
        raise


def read_plan(root=ROOT):
    root = Path(root)
    if sha256(root/"plan.json")!=(root/"plan.sha256").read_text().strip():
        raise ValueError("Saved collection plan changed.")
    plan = previous_training.read_json(root/"plan.json")
    if plan["contract"]!=contract(plan["contract"]["up_style"]) or previous_training.read_json(root/"contract.json")!=plan["contract"]:
        raise ValueError("V3 gesture contract differs from prepared definitions.")
    if plan["tasks"]!={str(s):tasks(plan["contract"]["up_style"],s) for s in (1,2)}:
        raise ValueError("Prepared task schedule changed.")
    for name,digest in plan["source_sha256"].items():
        if sha256(PROJECT/name)!=digest:
            raise ValueError("Prepared collection/features code changed.")
    for row in plan["imports"]:
        if sha256(row["file"])!=row["sha256"] or sha256(row["origin_file"])!=row["origin_sha256"]:
            raise ValueError("Reused recording or preserved original changed.")
    return plan


def audit(root=ROOT):
    root = Path(root);plan=read_plan(root)
    slots = {1:{},2:{}}; sessions={}; identities=set(plan["banned_raw_contents"])
    for path in sorted((root/"new_training").glob("*.npz")):
        # Reject an evaluation/foreign file before touching its coordinates.
        with np.load(path,allow_pickle=False) as a:
            meta=json.loads(str(a["metadata"]))
        s,i=meta.get("collection_sitting"),meta.get("task_index")
        if s not in slots or type(i) is not int or i not in range(25):
            raise ValueError("Foreign training metadata; no payload accessed.")
        required={"split":"train","label":"UP","source":"recording","taxonomy":TAXONOMY,
            "schema_version":VERSION,"feature_version":1,"labels":list(v2.LABELS),"convention":v2.CONVENTION,
            "command_contract":plan["contract"],"experiment_id":EXPERIMENT,"plan_sha256":sha256(root/"plan.json"),
            "task_hint":plan["tasks"][str(s)][i]["hint"],"up_shape":plan["tasks"][str(s)][i]["up_shape"]}
        if any(meta.get(k)!=v for k,v in required.items()) or any(meta.get(k) is not None for k in ("model","prediction","scores")):
            raise ValueError("Only fresh human-labeled replacement UP training allowed; no payload accessed.")
        session=meta.get("session")
        if not isinstance(session,str) or not session or session in plan["banned_sessions"]:
            raise ValueError("Invalid/historical training session.")
        created=datetime.fromisoformat(meta.get("created_utc",""))
        if created.tzinfo is None or created<=datetime.fromisoformat(plan["created_utc"]) or created>datetime.now(timezone.utc):
            raise ValueError("Replacement UP must be recorded after preparation.")
        if (session in sessions and sessions[session]!=s) or i in slots[s]:
            raise ValueError("Capture session crosses sittings or task repeated.")
        with np.load(path,allow_pickle=False) as a:
            xyz,times,visible=v2.validate_clip(a["xyz"],a["times"],a["visible"])
        identity=v2.content_sha(xyz,times,visible,meta["aspect"])
        if identity in identities or not v2.quality(times,visible):
            raise ValueError("Replacement UP is duplicate or has insufficient tracking.")
        identities.add(identity);sessions[session]=s;slots[s][i]=str(path.resolve())
    for s,assigned in slots.items():
        if set(assigned)!=set(range(len(assigned))):
            raise ValueError("Saved UP tasks must form a prefix without gaps.")
    if slots[2] and len(slots[1])!=25:
        raise ValueError("Finish sitting1 first.")
    return {"base_counts":plan["base_counts"],"old_up_in_active_training":0,"stop_ready":plan["stop_ready"],
            "sittings":{str(s):{"saved":len(slots[s]),"required":25,
                "prompted_shape_counts":dict(Counter(plan["tasks"][str(s)][i]["up_shape"] for i in slots[s]))} for s in (1,2)},
            "new_up_total":sum(len(a) for a in slots.values()),"up_collection_complete":all(len(a)==25 for a in slots.values()),
            "training_ready":False,"reason":"Fresh v3 validation and a separate fitting plan still required."}


class PromptKeys(dict):
    def __init__(self,current):
        super().__init__({"u":"UP"});self.current=current

    def __iter__(self):
        return iter(("u",)) if self.current() is not None else iter(())


def reject_model(path):
    raise ValueError("No compatible v3 model has been trained; this is a collection-only workflow.")


def record(sitting,root=ROOT,camera=0,hand="unspecified"):
    if sitting not in (1,2):
        raise ValueError("Use sitting1 or2.")
    root=Path(root);plan=read_plan(root);state=audit(root)
    sitting_tasks=plan["tasks"][str(sitting)]
    if sitting==2 and state["sittings"]["1"]["saved"]!=25:
        raise ValueError("Complete sitting1 before starting a new sitting2.")
    saved=initial=state["sittings"][str(sitting)]["saved"]
    if saved==25:
        print("This sitting is complete; no camera opened.");return 0
    attempt,started,clock=uuid4().hex,v2.utc_now(),time.monotonic()

    def current():
        return sitting_tasks[saved] if saved<25 else None

    def save(folder,xyz,times,visible,meta):
        nonlocal saved
        task=current()
        if task is None or meta.get("label")!="UP" or meta.get("split")!="train" or meta.get("source")!="recording" or any(meta.get(k) is not None for k in ("model","prediction","scores")):
            raise ValueError("Save only the current human-labeled UP training prompt.")
        if not v2.quality(times,visible):
            raise ValueError("Insufficient tracking; repeat the UP clip.")
        path=save_clip(folder,xyz,times,visible,{**meta,"experiment_id":EXPERIMENT,"collection_sitting":sitting,
            "task_index":saved,"task_hint":task["hint"],"up_shape":task["up_shape"],
            "plan_sha256":sha256(root/"plan.json"),"recording_attempt":attempt},plan["contract"])
        saved+=1
        print("25/25 saved. Press q." if current() is None else f"Next {saved+1}/25: {current()['hint']}",flush=True)
        return path

    from gesture_commands import app
    original=app.overlay

    def overlay(cv2,frame,lines):
        task=current()
        messages=["25/25 UP clips saved. Press q."] if task is None else [lines[0],
            f"V3 TRAIN {saved+1}/25 | replacement UP | save: u",*textwrap.wrap(task["hint"],65),
            "SPACE replay | u saves | x repeats | q pauses"]
        return original(cv2,frame,messages)

    profile=SimpleNamespace(data=root/"new_training",keys=PromptKeys(current),labels=v2.LABELS,save_clip=save,
        gate=v2.CommandGate,load_model=reject_model,window="Replacement UP - v3 training collection",
        metadata={"acquisition":"v3_replacement_up","hand_reported":hand})
    print(f"Sitting {sitting}, next {saved+1}/25: {current()['hint']}")
    print("Perform a NEW two-second gesture; SPACE replay, u save, x repeat, q pause. Resume only this physical sitting.")
    app.overlay,outcome=overlay,"failed"
    try:
        result=app.main(profile=profile,argv=["--split","train","--camera",str(camera)])
        outcome="complete" if saved==25 else "paused";return result
    finally:
        app.overlay=original
        (root/"recording_attempts").mkdir(exist_ok=True)
        write_json(root/"recording_attempts"/f"{attempt}.json",{"started_utc":started,"ended_utc":v2.utc_now(),
            "wall_seconds":time.monotonic()-clock,"sitting":sitting,"saved_before":initial,"saved_after":saved,
            "state":outcome,"scope":"camera/replay/preparation/pause interval; not pure annotation time"})


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="action",required=True)
    for action in ("prepare","status","record"):
        p=sub.add_parser(action);p.add_argument("--root",type=Path,default=ROOT)
        if action=="prepare":
            p.add_argument("--up-style",choices=tuple(UP_STYLES),required=True)
            p.add_argument("--stop-history",choices=("open_palm","mixed","closed_fist"),required=True)
            p.add_argument("--stop-history-notes",default=None)
            p.add_argument("--study",type=Path,default=STUDY)
        if action=="record":
            p.add_argument("--sitting",type=int,choices=(1,2),required=True)
            p.add_argument("--camera",type=int,default=0);p.add_argument("--hand",default="unspecified")
    a=parser.parse_args()
    if a.action=="prepare":prepare(a.up_style,a.stop_history,a.root,a.study,stop_history_notes=a.stop_history_notes)
    elif a.action=="record":record(a.sitting,a.root,a.camera,a.hand)
    else:print(json.dumps(audit(a.root),indent=2))


if __name__=="__main__":
    main()
