"""Freeze experiment011; collect two fresh sittings and jointly evaluate once."""

import argparse
import json
import shutil
import textwrap
import time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import torch

from . import cross_model_motion as experiment
from . import commands_v3_validation as validation
from .run import PROJECT,sha256,write_json

ROOT=PROJECT/"research/experiments/data/prospective_motion_011"
EXPERIMENT="012_prospective_motion_sessions"
KEY=validation.KEY
SOURCES=tuple(dict.fromkeys((*experiment.SOURCES,"research/experiments/prospective_motion.py",
    "research/experiments/test_prospective_motion.py","research/PROSPECTIVE_MOTION_PLAN.md")))


def schedule(contract,sitting):
    if sitting not in (1,2):raise ValueError("Use sitting1 or2.")
    tasks=validation.schedule(contract)
    if sitting==2:
        tasks=[task for i in range(0,48,8) for task in reversed(tasks[i:i+8])]
    return [{**task,"index":i} for i,task in enumerate(tasks)]


def freeze(study,root=ROOT):
    study,root=Path(study),Path(root)
    if root.exists() and any(root.iterdir()):raise ValueError("Freeze destination exists; preserve it.")
    config=experiment.motion.previous.fitting.read_json(study/"config.json")
    required=len(experiment.jobs(config,"core"))+len(experiment.jobs(config,"controls"))
    if experiment.motion.previous.fitting.read_json(study/"status.json")!={"state":"controls_complete","completed_trials":required}:
        raise ValueError("Complete both core and control phases before freezing.")
    checked=experiment.verify(study)
    environment=experiment.motion.previous.fitting.read_json(study/"environment.json")
    if any(sha256(PROJECT/name)!=digest for name,digest in environment["source_sha256"].items() if name.endswith(".py")):
        raise ValueError("Current executable sources differ from the fitted study.")
    audit=experiment.motion.previous.fitting.read_json(study/"dataset_audit.json")
    if audit["malformed_records"] or audit["conflicting_canonical_label_groups"]:raise ValueError("Resolve audit conflicts first.")
    manifest=experiment.motion.previous.fitting.read_json(study/"manifest.json")
    trials=experiment.motion.previous.fitting.read_json(study/"trials.json")
    plan={"experiment_id":EXPERIMENT,"created_utc":datetime.now(timezone.utc).isoformat(),"study":str(study.resolve()),
        "contract":manifest["contract"],"participant":"current_user","required_sittings":2,"clips_per_sitting":48,
        "schedules":{str(s):schedule(manifest["contract"],s) for s in (1,2)},
        "checkpoints":[{"folder":experiment.folder_name(r["family"],r["mode"],r["variant"],r["seed"]),
            "family":r["family"],"mode":r["mode"],"variant":r["variant"],"seed":r["seed"],"stage":r["stage"],
            "sha256":r["checkpoint_sha256"]} for r in trials],
        "study_sha256":{name:sha256(study/name) for name in ("config.json","manifest.json","trials.json","summary.json","status.json","dataset_audit.json")},
        "banned_sessions":sorted({r["session"] for r in audit["records"]}),
        "banned_raw_contents":sorted({r["raw_content_sha256"] for r in audit["records"]}),
        "source_sha256":{name:sha256(PROJECT/name) for name in SOURCES},
        "scope":"All candidates, one joint fresh same-user development-session check; no retraining/gates/early-response/final-paper claim."}
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/"freeze.json",plan);(root/"freeze.sha256").write_text(sha256(root/"freeze.json")+"\n")
    for name in SOURCES:
        target=root/"source"/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/name,target)
    write_json(root/"preparation_verification.json",checked)
    print(f"Frozen {len(trials)} models. Collect two new48-clip sittings; no camera opened.")
    return plan


def read_plan(root=ROOT):
    root=Path(root)
    if sha256(root/"freeze.json")!=(root/"freeze.sha256").read_text().strip():raise ValueError("Frozen policy changed.")
    plan=experiment.motion.previous.fitting.read_json(root/"freeze.json")
    study=Path(plan["study"])
    for name,digest in plan["study_sha256"].items():
        if sha256(study/name)!=digest:raise ValueError("Frozen study changed.")
    for name,digest in plan["source_sha256"].items():
        if sha256(root/"source"/name)!=digest or (name.endswith(".py") and sha256(PROJECT/name)!=digest):raise ValueError("Frozen executable source changed.")
    for row in plan["checkpoints"]:
        if sha256(study/row["folder"]/"model.pt")!=row["sha256"]:raise ValueError("Frozen model changed.")
    if plan["schedules"]!={str(s):schedule(plan["contract"],s) for s in (1,2)}:raise ValueError("Schedule changed.")
    return plan


def status(root=ROOT):
    """Metadata only: never decode raw held-out arrays while collection is incomplete."""
    root=Path(root);plan=read_plan(root);rows=[];seen=set(plan["banned_raw_contents"]);sessions={};slots={1:set(),2:set()}
    for path in sorted((root/"recordings").glob("*.npz")):
        with np.load(path,allow_pickle=False) as a:meta=json.loads(str(a["metadata"]))
        sitting,index=meta.get("prospective_sitting"),meta.get("task_index")
        if sitting not in (1,2) or type(index)is not int or index not in range(48):raise ValueError("Foreign cohort metadata; no payload accessed.")
        task=plan["schedules"][str(sitting)][index]
        expected={"schema_version":3,"feature_version":1,"taxonomy":validation.v3.TAXONOMY,
            "experiment_id":EXPERIMENT,"split":"validation","label":task["label"],"command_contract":plan["contract"],
            "freeze_sha256":sha256(root/"freeze.json"),"up_shape":task["up_shape"],"task_hint":task["hint"],"source":"recording"}
        if any(meta.get(k)!=v for k,v in expected.items()) or any(meta.get(k)is not None for k in ("model","prediction","scores")):
            raise ValueError("Foreign/model-assisted prospective record; no payload accessed.")
        identity=meta.get("raw_content_sha256");session=meta.get("session")
        if not isinstance(identity,str) or identity in seen or not session or session in plan["banned_sessions"] or index in slots[sitting]:
            raise ValueError("Duplicate/historical/session reuse in prospective metadata.")
        if session in sessions and sessions[session]!=sitting:raise ValueError("Capture session crosses physical sittings.")
        if meta.get("passes_tracking_quality") is not True and task["label"]!="NO_COMMAND":raise ValueError("Command tracking insufficient.")
        seen.add(identity);sessions[session]=sitting;slots[sitting].add(index)
        rows.append({"file":str(path.resolve()),"sha256":sha256(path),"label":task["label"],"session":session,"sitting":sitting,
            "task_index":index,"raw_content_sha256":identity,"passes_tracking_quality":meta["passes_tracking_quality"]})
    if any(values!=set(range(len(values))) for values in slots.values()):raise ValueError("Saved tasks must form a prefix per sitting.")
    return {"counts":{str(s):len(values) for s,values in slots.items()},"complete":all(len(v)==48 for v in slots.values()),
        "records":sorted(rows,key=lambda r:(r["sitting"],r["task_index"])),"human_labels":len(rows)}


def record(sitting,root=ROOT,camera=0):
    if sitting not in (1,2):raise ValueError("Use sitting1 or2.")
    root=Path(root);plan=read_plan(root);state=status(root);tasks=plan["schedules"][str(sitting)]
    if sitting==2 and state["counts"]["1"]!=48:raise ValueError("Complete sitting1 before a new sitting2.")
    saved=initial=state["counts"][str(sitting)]
    if saved==48:print("This sitting is complete; no camera opened.");return 0
    seen=set(plan["banned_raw_contents"])|{r["raw_content_sha256"] for r in state["records"]}
    forbidden_sessions=set(plan["banned_sessions"])|{r["session"] for r in state["records"] if r["sitting"]!=sitting}
    attempt=uuid4().hex;started=time.monotonic();started_utc=validation.v2.utc_now()
    def current():return tasks[saved] if saved<48 else None
    def save(folder,xyz,times,visible,meta):
        nonlocal saved
        task=current()
        if (task is None or meta.get("label")!=task["label"] or meta.get("split")!="validation" or meta.get("source")!="recording"
            or any(meta.get(k)is not None for k in ("model","prediction","scores"))):raise ValueError("Save only the prompted human label.")
        if meta.get("session") in forbidden_sessions:raise ValueError("New physical sitting cannot share old session.")
        xyz,times,visible=validation.v2.validate_clip(xyz,times,visible)
        quality=bool(validation.v2.quality(times,visible))
        if not quality and meta["label"]!="NO_COMMAND":raise ValueError("Repeat poorly tracked command.")
        identity=validation.v2.content_sha(xyz,times,visible,meta["aspect"])
        if identity in seen:raise ValueError("Perform a genuinely new recording.")
        path=validation.v3.save_clip(folder,xyz,times,visible,{**meta,"experiment_id":EXPERIMENT,"prospective_sitting":sitting,
            "task_index":saved,"up_shape":task["up_shape"],"task_hint":task["hint"],"freeze_sha256":sha256(root/"freeze.json"),
            "raw_content_sha256":identity,"passes_tracking_quality":quality,"recording_attempt":attempt},plan["contract"])
        seen.add(identity);saved+=1
        print("48/48 complete. Press q." if current()is None else f"Next {saved+1}/48: {current()['label']} — {current()['hint']}",flush=True)
        return path
    from gesture_commands import app
    original=app.overlay
    def overlay(cv2,frame,lines):
        task=current();messages=["48/48 saved. Press q."] if task is None else [lines[0],f"FRESH SITTING {sitting}: {saved+1}/48 | {task['label']} | save {KEY[task['label']]}",
            *textwrap.wrap(task["hint"],65),"SPACE replay | prompted key saves | x repeats | q pauses"]
        return original(cv2,frame,messages)
    profile=SimpleNamespace(data=root/"recordings",keys=validation.PromptKeys(current),labels=validation.v2.LABELS,save_clip=save,
        gate=validation.v2.CommandGate,load_model=validation.v3.reject_model,window="Fresh motion validation - no guesses",
        metadata={"acquisition":"prospective_session_check"})
    print(f"Next {saved+1}/48: {current()['label']} — {current()['hint']}")
    print("Use a NEW sitting; q preserves progress. Resume only the same physical sitting. No model guesses loaded.")
    outcome="failed";app.overlay=overlay
    try:
        result=app.main(profile=profile,argv=["--split","validation","--camera",str(camera)])
        outcome="complete" if saved==48 else "paused";return result
    finally:
        app.overlay=original;(root/"attempts").mkdir(exist_ok=True)
        write_json(root/"attempts"/f"{attempt}.json",{"sitting":sitting,"started_utc":started_utc,"ended_utc":validation.v2.utc_now(),
            "wall_seconds":time.monotonic()-started,"saved_before":initial,"saved_after":saved,"state":outcome,
            "scope":"Camera/replay/preparation/pause interval, not pure annotation time."})


def evaluate(root=ROOT):
    root=Path(root);plan=read_plan(root);state=status(root)
    if not state["complete"]:raise ValueError("Complete both48-clip sittings before any performance evaluation.")
    access=root/"evaluation_access.json"
    if access.exists():raise ValueError("Pool already accessed/retired. Preserve prior evaluation; verification uses saved arrays.")
    write_json(access,{"state":"access_started","started_utc":validation.v2.utc_now(),"scope":"Any payload access retires this pool as inspected."})
    output=root/"evaluation";output.mkdir(exist_ok=False)
    try:
        raw=[]
        for row in state["records"]:
            if sha256(row["file"])!=row["sha256"]:raise ValueError("Prospective raw changed.")
            xyz,times,visible,meta=experiment.motion.previous.training.load_clip(row["file"],plan["contract"])
            identity=validation.v2.content_sha(xyz,times,visible,meta["aspect"])
            if identity!=row["raw_content_sha256"]:raise ValueError("Raw identity differs from acquisition metadata.")
            raw.append((xyz,times,visible,meta))
        arrays={mode:np.stack([experiment.motion.rep.features(x,t,v,m["aspect"],mode)[0] for x,t,v,m in raw]) for mode in experiment.motion.rep.MODES}
        if any(sha256(r["file"])!=r["sha256"] for r in state["records"]):raise ValueError("Raw changed during extraction.")
        y=np.array([validation.v2.LABELS.index(r["label"]) for r in state["records"]],dtype=np.int64)
        np.savez_compressed(output/"features.npz",y=y,**arrays);write_json(output/"manifest.json",state)
        torch.set_num_threads(2);trials=[]
        for candidate in plan["checkpoints"]:
            model,_=experiment.load_model(Path(plan["study"])/candidate["folder"]/"model.pt")
            x=experiment.inputs(arrays[candidate["mode"]],candidate["family"],candidate["variant"])
            with torch.inference_mode():p=model(torch.from_numpy(x)).softmax(1).numpy()
            samples=list(zip(x,y,[r["file"] for r in state["records"]]))
            folder=output/candidate["folder"];folder.mkdir()
            write_json(folder/"predictions.json",validation.v2.predictions(samples,p))
            trial={**candidate,"pooled":experiment.motion.previous.fitting.metric_set(p,y),"sittings":{}}
            for sitting in (1,2):
                indices=[i for i,r in enumerate(state["records"]) if r["sitting"]==sitting]
                trial["sittings"][str(sitting)]=experiment.motion.previous.fitting.metric_set(p[indices],y[indices])
            write_json(folder/"metrics.json",trial);trials.append(trial)
        write_json(output/"trials.json",trials)
        write_json(output/"verification.json",verify_evaluation(root))
        write_json(access,{"state":"complete","ended_utc":validation.v2.utc_now(),"models":len(trials),"clips":len(y),
            "scope":"Joint prospective same-user development check now inspected; no future untouched reuse."})
    except BaseException as e:
        write_json(access,{"state":"failed_retired","ended_utc":validation.v2.utc_now(),"error":str(e)});raise
    print(f"Joint scores saved: {output}. This pool is now inspected.")


def verify_evaluation(root=ROOT):
    root=Path(root);plan=read_plan(root);output=root/"evaluation"
    manifest=experiment.motion.previous.fitting.read_json(output/"manifest.json")
    trials=experiment.motion.previous.fitting.read_json(output/"trials.json")
    if len(trials)!=len(plan["checkpoints"]):raise ValueError("Incomplete candidate scores.")
    if any(sha256(r["file"])!=r["sha256"] for r in manifest["records"]):raise ValueError("Original prospective recording changed.")
    count=0
    with np.load(output/"features.npz",allow_pickle=False) as a:
        for candidate,trial in zip(plan["checkpoints"],trials):
            if any(trial.get(k)!=v for k,v in candidate.items()):raise ValueError("Candidate membership/order differs.")
            model,_=experiment.load_model(Path(plan["study"])/candidate["folder"]/"model.pt")
            x=experiment.inputs(a[candidate["mode"]],candidate["family"],candidate["variant"])
            with torch.inference_mode():p=model(torch.from_numpy(x)).softmax(1).numpy()
            rows=experiment.motion.previous.fitting.read_json(output/candidate["folder"]/"predictions.json")
            if [r["file"] for r in rows]!=[r["file"] for r in manifest["records"]] or [r["actual"] for r in rows]!=[validation.v2.LABELS[int(i)] for i in a["y"]]:raise ValueError("Prediction rows changed.")
            np.testing.assert_allclose(p,[r["scores"] for r in rows],rtol=1e-6,atol=1e-7)
            if [r["predicted"] for r in rows]!=[validation.v2.LABELS[int(i)] for i in p.argmax(1)]:raise ValueError("Predicted names changed.")
            if experiment.motion.previous.fitting.read_json(output/candidate["folder"]/"metrics.json")!=trial:raise ValueError("Trial copies differ.")
            if experiment.motion.previous.fitting.metric_set(p,a["y"])!=trial["pooled"]:raise ValueError("Pooled metrics differ.")
            for sitting in (1,2):
                indices=[i for i,r in enumerate(manifest["records"]) if r["sitting"]==sitting]
                if experiment.motion.previous.fitting.metric_set(p[indices],a["y"][indices])!=trial["sittings"][str(sitting)]:raise ValueError("Sitting metrics differ.")
            count+=len(rows)
    return {"prediction_vectors_verified":count,"models":len(trials),"scope":"Artifact integrity of an already inspected pool."}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    f=sub.add_parser("freeze");f.add_argument("study",type=Path)
    s=sub.add_parser("record");s.add_argument("--sitting",type=int,choices=(1,2),required=True);s.add_argument("--camera",type=int,default=0)
    for name in ("status","evaluate","verify"):sub.add_parser(name)
    args=p.parse_args()
    if args.action=="freeze":freeze(args.study)
    elif args.action=="record":record(args.sitting,camera=args.camera)
    elif args.action=="status":print(json.dumps({k:v for k,v in status().items() if k!="records"},indent=2))
    elif args.action=="evaluate":evaluate()
    else:print(verify_evaluation())


if __name__=="__main__":main()
