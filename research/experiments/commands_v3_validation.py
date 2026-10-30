"""Guided fresh v3 validation: 48 clips kept outside training weight updates.

Archives the training cohort and collection rules before human recording.
Reuses existing camera/replay/landmark code without modifying frozen sources.
No model predictions, fitting, test access or automatic checkpoint activation.
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

from . import commands_v3_collection as v3
from . import commands_v2 as v2
from .boundary_train import read_json
from .run import PROJECT, sha256, write_json

EXPERIMENT = "006_v3_validation"
KEY = {label: key for key,label in v2.KEYS.items()}
SOURCES = (*v3.SOURCES, "research/experiments/commands_v3_validation.py",
           "research/experiments/test_commands_v3_validation.py")
HINTS = {
    "FORWARD": ("Use your comfortable come-toward-you beckoning.", "Beckon with one finger.",
                "Beckon with several fingers.", "Beckon with an open hand.",
                "Beckon at a modestly different angle.", "Beckon at a different comfortable distance."),
    "BACKWARD": ("Use your comfortable outward push-away gesture.", "Push away with one finger extended.",
                 "Push away with several fingers extended.", "Push away with an open hand.",
                 "Push away at a modestly different angle.", "Push away at a different comfortable distance."),
    "LEFT": ("Point left in the mirrored preview using one finger.", "Point left using several fingers.",
             "Point left in your usual comfortable style.", "Point left at a modestly different angle.",
             "Point left at a different comfortable height.", "Point left at a different comfortable distance."),
    "RIGHT": ("Point right in the mirrored preview using one finger.", "Point right using several fingers.",
              "Point right in your usual comfortable style.", "Point right at a modestly different angle.",
              "Point right at a different comfortable height.", "Point right at a different comfortable distance."),
    "STOP": ("Hold a still open palm facing the camera, all fingers extended.",
             "Hold the same open-palm STOP at a slightly different height.",
             "Hold the same open-palm STOP at a comfortable nearer distance.",
             "Hold the same open-palm STOP at a modest angle, palm still visible.",
             "Hold the same open-palm STOP at a comfortable farther distance.",
             "Hold a still open palm facing the camera, all fingers extended."),
    "NO_COMMAND": ("Rest your visible hand naturally; no intended command.",
                   "Keep a naturally lowered relaxed hand at rest; no command.",
                   "Reposition your relaxed hand naturally, without a command.",
                   "Rest with relaxed fingers facing down; do not deliberately point DOWN.",
                   "Rest your relaxed visible hand at another comfortable height; no command.",
                   "Keep both hands out of view for about two seconds; no command."),
    "DOWN": ("Deliberately point one finger downward.", "Deliberately point several fingers downward.",
             "Deliberately move the whole hand downward to request DOWN.",
             "Deliberately point down at a modestly different angle.",
             "Deliberately move the hand downward at a different comfortable distance.",
             "Use your usual clear, deliberate DOWN gesture."),
}


def schedule(contract):
    tasks=[]
    shapes=contract["allowed_up_shapes"]
    for repeat in range(6):
        for label in v2.LABELS:
            shape=shapes[repeat%len(shapes)] if label=="UP" else None
            hint=(v3.UP_STYLES[shape]+(" Use a comfortable central position." if repeat<3
                  else " Use a modestly different height, angle or distance; keep the indicating finger(s) upward.")) if shape else HINTS[label][repeat]
            tasks.append({"index":len(tasks),"label":label,"up_shape":shape,"hint":hint})
    return tasks


def training_snapshot(root):
    """Read only authorized training folders; never old validation/test payloads."""
    root=Path(root);plan=v3.read_plan(root);state=v3.audit(root)
    if not state["up_collection_complete"]:
        raise ValueError("Complete all50 new UP training clips before validation preparation.")
    files=sorted((root/"reused_training").glob("*.npz"))+sorted((root/"new_training").glob("*.npz"))
    if {str(p.resolve()) for p in (root/"reused_training").glob("*.npz")}!={r["file"] for r in plan["imports"]}:
        raise ValueError("Reused training folder contains undeclared files.")
    metadata=[]
    for p in files:
        with np.load(p,allow_pickle=False) as a:
            m=json.loads(str(a["metadata"]))
        if m.get("split")!="train" or m.get("command_contract")!=plan["contract"] or m.get("schema_version")!=3:
            raise ValueError("Foreign/non-training data in training snapshot; no foreign payload read.")
        metadata.append(m)
    records=[];seen=set()
    for path,m in zip(files,metadata):
        digest=sha256(path)
        with np.load(path,allow_pickle=False) as a:
            xyz,times,visible=v2.validate_clip(a["xyz"],a["times"],a["visible"])
        identity=v2.content_sha(xyz,times,visible,m["aspect"])
        if identity in seen:
            raise ValueError("Raw duplicate in training snapshot.")
        seen.add(identity)
        if sha256(path)!=digest:
            raise ValueError("Training recording changed during snapshot.")
        records.append({"file":str(path.resolve()),"sha256":digest,"label":m["label"],"session":m["session"],
                        "raw_content_sha256":identity,"passes_tracking_quality":v2.quality(times,visible)})
    if len(records)!=len(plan["imports"])+50:
        raise ValueError("Training cohort size differs from declared replacement collection.")
    return plan,records


def prepare(root=v3.ROOT,legacy_migration=v2.DATA/"migration.json"):
    root=Path(root)
    if (root/"validation_plan.json").exists() or any((root/"validation").glob("*.npz")):
        raise ValueError("Validation plan/data already exists; preserve it and resume instead.")
    source,records=training_snapshot(root)
    sessions=set(source["banned_sessions"])|{r["session"] for r in records}
    identities=set(source["banned_raw_contents"])|{r["raw_content_sha256"] for r in records}
    if legacy_migration is not None:
        historical=read_json(legacy_migration)
        sessions.update(historical["historical_sessions"])
        identities.update(historical["historical_development_raw_hashes"])
    plan={"experiment_id":EXPERIMENT,"created_utc":v2.utc_now(),"contract":source["contract"],
          "training_plan_sha256":sha256(root/"plan.json"),"training_records":records,
          "training_counts":dict(Counter(r["label"] for r in records)),"tasks":schedule(source["contract"]),
          "required_clips":48,"physical_sittings":1,"banned_sessions":sorted(sessions),
          "banned_raw_contents":sorted(identities),"source_sha256":{s:sha256(PROJECT/s) for s in SOURCES},
          "rule":"Fresh human-only validation, all six/class. No predictions during labeling, weight updates, test claims or automatic camera checkpoint activation.",
          "purpose":"Checkpoint selection and development diagnosis; validation outcomes are inspected development, not an untouched final benchmark.",
          "annotation_cost":"48 new development-evaluation labels; not additional training labels."}
    for name in SOURCES:
        destination=root/"validation_source"/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(PROJECT/name,destination)
    write_json(root/"validation_plan.json",plan)
    (root/"validation_plan.sha256").write_text(sha256(root/"validation_plan.json")+"\n")
    print(f"Prepared48 fresh validation prompts; {len(records)} training clips preserved outside validation.")
    return plan


def read_plan(root=v3.ROOT):
    root=Path(root)
    if sha256(root/"validation_plan.json")!=(root/"validation_plan.sha256").read_text().strip():
        raise ValueError("Validation plan changed.")
    plan=read_json(root/"validation_plan.json");source=v3.read_plan(root)
    if sha256(root/"plan.json")!=plan["training_plan_sha256"] or plan["contract"]!=source["contract"] or plan["tasks"]!=schedule(source["contract"]):
        raise ValueError("Prepared training contract or validation schedule differs.")
    for name,digest in plan["source_sha256"].items():
        if sha256(PROJECT/name)!=digest or sha256(root/"validation_source"/name)!=digest:
            raise ValueError("Prepared validation/source code changed.")
    actual={str(p.resolve()) for folder in ("reused_training","new_training") for p in (root/folder).glob("*.npz")}
    if actual!={r["file"] for r in plan["training_records"]}:
        raise ValueError("Training cohort changed after validation preparation.")
    for r in plan["training_records"]:
        if sha256(r["file"])!=r["sha256"]:
            raise ValueError("Training recording changed after validation preparation.")
    return plan


def audit(root=v3.ROOT):
    root=Path(root);plan=read_plan(root);slots={};seen=set(plan["banned_raw_contents"]);rows=[]
    for path in sorted((root/"validation").glob("*.npz")):
        with np.load(path,allow_pickle=False) as a:m=json.loads(str(a["metadata"]))
        i=m.get("task_index")
        if type(i) is not int or i not in range(48):
            raise ValueError("Foreign validation metadata; no payload accessed.")
        task=plan["tasks"][i]
        expected={"split":"validation","source":"recording","label":task["label"],"task_hint":task["hint"],
            "up_shape":task["up_shape"],"schema_version":3,"feature_version":1,"taxonomy":v3.TAXONOMY,
            "convention":v2.CONVENTION,"labels":list(v2.LABELS),"command_contract":plan["contract"],
            "experiment_id":EXPERIMENT,"validation_plan_sha256":sha256(root/"validation_plan.json"),"collection_sitting":1}
        if any(m.get(k)!=value for k,value in expected.items()) or any(m.get(k) is not None for k in ("model","prediction","scores")):
            raise ValueError("Validation violates human-only role/contract; no payload accessed.")
        session=m.get("session")
        if not isinstance(session,str) or not session or session in plan["banned_sessions"]:
            raise ValueError("Validation session overlaps training/history.")
        created=datetime.fromisoformat(m.get("created_utc",""))
        if created.tzinfo is None or created<=datetime.fromisoformat(plan["created_utc"]) or created>datetime.now(timezone.utc):
            raise ValueError("Validation must be recorded after preparation.")
        if i in slots:
            raise ValueError("Repeated validation task.")
        digest=sha256(path)
        with np.load(path,allow_pickle=False) as a:xyz,times,visible=v2.validate_clip(a["xyz"],a["times"],a["visible"])
        if not np.isfinite(m.get("aspect",np.nan)) or m["aspect"]<=0:
            raise ValueError("Invalid validation camera aspect.")
        identity=v2.content_sha(xyz,times,visible,m["aspect"])
        if identity in seen:
            raise ValueError("Validation raw contents duplicate training/history/another validation clip.")
        seen.add(identity);tracked=v2.quality(times,visible)
        if not tracked and m["label"]!="NO_COMMAND":
            raise ValueError("Command validation clip has insufficient tracking.")
        if sha256(path)!=digest:
            raise ValueError("Validation recording changed during audit.")
        row={"file":str(path.resolve()),"sha256":digest,"session":session,"label":m["label"],
             "task_index":i,"up_shape":task["up_shape"],"raw_content_sha256":identity,"passes_tracking_quality":tracked}
        rows.append(row);slots[i]=row
    if set(slots)!=set(range(len(slots))):
        raise ValueError("Validation tasks must form a saved prefix without gaps.")
    return {"saved":len(rows),"required":48,"complete":len(rows)==48,
            "labels":dict(Counter(r["label"] for r in rows)),
            "prompted_up_shapes":dict(Counter(r["up_shape"] for r in rows if r["up_shape"])),
            "quality_failures_retained":sum(not r["passes_tracking_quality"] for r in rows),
            "training_clips":len(plan["training_records"]),"records":sorted(rows,key=lambda r:r["task_index"])}


class PromptKeys(dict):
    def __init__(self,current):
        super().__init__(v2.KEYS);self.current=current

    def __iter__(self):
        task=self.current()
        return iter((KEY[task["label"]],)) if task is not None else iter(())


def record(root=v3.ROOT,camera=0,hand="unspecified"):
    root=Path(root);plan=read_plan(root);state=audit(root)
    saved=initial=state["saved"]
    if state["complete"]:
        print("48/48 validation complete; no camera opened.");return 0
    seen=set(plan["banned_raw_contents"])|{r["raw_content_sha256"] for r in state["records"]}
    attempt,started,clock=uuid4().hex,v2.utc_now(),time.monotonic()

    def current():return plan["tasks"][saved] if saved<48 else None

    def save(folder,xyz,times,visible,meta):
        nonlocal saved
        task=current()
        if task is None or meta.get("label")!=task["label"] or meta.get("split")!="validation" or meta.get("source")!="recording" or any(meta.get(k) is not None for k in ("model","prediction","scores")):
            raise ValueError("Save only the current human-labeled validation prompt.")
        if meta.get("session") in plan["banned_sessions"]:
            raise ValueError("Validation cannot share a historical/training session.")
        xyz,times,visible=v2.validate_clip(xyz,times,visible)
        if meta["label"]!="NO_COMMAND" and not v2.quality(times,visible):
            raise ValueError("Tracking insufficient; repeat this command clip.")
        identity=v2.content_sha(xyz,times,visible,meta["aspect"])
        if identity in seen:
            raise ValueError("Duplicate recording; perform a genuinely new validation clip.")
        path=v3.save_clip(folder,xyz,times,visible,{**meta,"experiment_id":EXPERIMENT,"collection_sitting":1,
            "task_index":saved,"up_shape":task["up_shape"],"task_hint":task["hint"],
            "validation_plan_sha256":sha256(root/"validation_plan.json"),"recording_attempt":attempt},plan["contract"])
        seen.add(identity);saved+=1
        print("48/48 complete. Press q." if current() is None else
              f"Next{saved+1}/48: {current()['label']} — {current()['hint']}",flush=True)
        return path

    from gesture_commands import app
    original=app.overlay

    def overlay(cv2,frame,lines):
        task=current()
        messages=["48/48 validation saved. Press q.","Training clips remain separate; no model loaded."] if task is None else [
            lines[0],f"V3 VALIDATION {saved+1}/48 | {task['label']} | save: {KEY[task['label']]}",
            *textwrap.wrap(task["hint"],65),"SPACE replay | prompted key saves | x repeats | q pauses"]
        return original(cv2,frame,messages)

    profile=SimpleNamespace(data=root/"validation",keys=PromptKeys(current),labels=v2.LABELS,save_clip=save,
        gate=v2.CommandGate,load_model=v3.reject_model,window="V3 validation - guided human labels",
        metadata={"hand_reported":hand,"acquisition":"fresh_v3_validation"})
    print(f"Next{saved+1}/48: {current()['label']} — {current()['hint']}")
    print("Start a NEW physical sitting, separate from training. No predictions shown. SPACE review; prompted key saves.")
    print("q pauses; restarting resumes the next prompt. Resume only within this same physical sitting.")
    app.overlay,outcome=overlay,"failed"
    try:
        result=app.main(profile=profile,argv=["--split","validation","--camera",str(camera)])
        outcome="complete" if saved==48 else "paused";return result
    finally:
        app.overlay=original;(root/"validation_attempts").mkdir(exist_ok=True)
        write_json(root/"validation_attempts"/f"{attempt}.json",{"started_utc":started,"ended_utc":v2.utc_now(),
            "wall_seconds":time.monotonic()-clock,"saved_before":initial,"saved_after":saved,"state":outcome,
            "scope":"camera/replay/preparation/pause interval; not pure annotation time"})


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="action",required=True)
    for action in ("prepare","status","record"):
        p=sub.add_parser(action);p.add_argument("--root",type=Path,default=v3.ROOT)
        if action=="record":p.add_argument("--camera",type=int,default=0);p.add_argument("--hand",default="unspecified")
    args=parser.parse_args()
    if args.action=="prepare":prepare(args.root)
    elif args.action=="record":record(args.root,args.camera,args.hand)
    else:print(json.dumps({k:v for k,v in audit(args.root).items() if k!="records"},indent=2))


if __name__=="__main__":main()
