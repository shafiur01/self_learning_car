"""Audit canonical raw datasets without rescoring historical test performance."""

import argparse
import json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from . import commands_v2 as v2
from . import cross_model_motion as experiment
from .run import PROJECT,sha256,write_json

LOCATIONS={
    "legacy":PROJECT/"gesture_commands/data",
    "v2":PROJECT/"research/experiments/data/commands_v2",
    "v3_reused_training":PROJECT/"research/experiments/data/commands_v3/reused_training",
    "v3_new_training":PROJECT/"research/experiments/data/commands_v3/new_training",
    "v3_validation":PROJECT/"research/experiments/data/commands_v3/validation",
    "boundary_training":PROJECT/"research/experiments/data/boundary_pilot/recordings",
    "boundary_test":PROJECT/"research/experiments/data/boundary_test/recordings"}


def read_record(path,cohort):
    before=sha256(path)
    with np.load(path,allow_pickle=False) as a:
        meta=json.loads(str(a["metadata"]))
        if (meta.get("schema_version") not in (1,2,3) or meta.get("convention")!=v2.CONVENTION
            or meta.get("split") not in ("train","validation","test") or meta.get("label") not in v2.LABELS
            or not isinstance(meta.get("session"),str) or not meta["session"]):
            raise ValueError("Unsupported role/schema/convention/label/session.")
        xyz,times,visible=v2.validate_clip(a["xyz"],a["times"],a["visible"])
    if not np.isfinite(meta.get("aspect",np.nan)) or meta["aspect"]<=0:raise ValueError("Invalid aspect ratio.")
    if before!=sha256(path):raise ValueError("Raw recording changed during audit.")
    return {"file":str(path.resolve()),"sha256":before,"cohort":cohort,"split":meta["split"],"label":meta["label"],
        "session":meta["session"],"raw_content_sha256":v2.content_sha(xyz,times,visible,meta["aspect"]),
        "schema_version":meta["schema_version"],"frames":len(times),"duration_seconds":float(times[-1]-times[0]),
        "tracking_fraction":float(visible.mean()),"passes_tracking_quality":bool(v2.quality(times,visible)),
        "sitting":meta.get("test_sitting",meta.get("collection_sitting")),
        "semantic_caveat":"Historical STOP labels rely on qualified open-palm report; not independently all verified."
            if meta["label"]=="STOP" and not cohort.startswith("v3_") else None}


def classify(row,training_ids,training_sessions):
    if row["split"]=="train":return "training_role"
    if row["raw_content_sha256"] in training_ids:return "raw_recording_in_current_training"
    if row["session"] in training_sessions:return "session_in_current_training"
    if row["label"]=="UP" and row["cohort"]!="v3_validation":return "old_up_definition"
    return "compatible_development_only" if row["cohort"]=="v3_validation" else "historical_common_label_only"


def audit(study):
    study=Path(study);manifest=experiment.motion.previous.fitting.read_json(study/"manifest.json")
    training=[r for r in manifest["records"] if r["split"]=="train"]
    train_ids={r["raw_content_sha256"] for r in training};train_sessions={r["session"] for r in training}
    rows=[];failures=[]
    for cohort,folder in LOCATIONS.items():
        for path in sorted(folder.glob("*.npz")):
            try:
                row=read_record(path,cohort);row["evaluation_status"]=classify(row,train_ids,train_sessions);rows.append(row)
            except Exception as e:failures.append({"file":str(path),"reason":str(e)})
    identities=defaultdict(list)
    for row in rows:identities[row["raw_content_sha256"]].append(row)
    duplicates=[{"raw_content_sha256":identity,"files":[r["file"] for r in items],"roles":sorted({r["split"] for r in items}),
        "labels":sorted({r["label"] for r in items})} for identity,items in identities.items() if len(items)>1]
    conflicts=[r for r in duplicates if len(r["labels"])>1]
    cohorts={}
    for cohort in LOCATIONS:
        items=[r for r in rows if r["cohort"]==cohort]
        cohorts[cohort]={"files":len(items),"sessions":len({r["session"] for r in items}),
            "roles":dict(Counter(r["split"] for r in items)),"labels":dict(Counter(r["label"] for r in items)),
            "evaluation_status":dict(Counter(r["evaluation_status"] for r in items)),
            "tracking_quality_failures":sum(not r["passes_tracking_quality"] for r in items),
            "no_hand_clips":sum(r["tracking_fraction"]==0 for r in items)}
    groups={}
    for cohort,role in (("legacy","validation"),("legacy","test"),("v2","validation"),("boundary_test","test"),("v3_validation","validation")):
        items=[r for r in rows if r["cohort"]==cohort and r["split"]==role]
        eligible=[r for r in items if r["evaluation_status"] in ("compatible_development_only","historical_common_label_only")]
        groups[cohort+"/"+role]={"total_files":len(items),"semantic_and_split_compatible":len(eligible),
            "exclusions":dict(Counter(r["evaluation_status"] for r in items if r not in eligible)),
            "compatible_labels":dict(Counter(r["label"] for r in eligible)),
            "performance_scored_now":False,"scope":"Checkpoint-selected development" if cohort=="v3_validation" else "Previously inspected; no new test performance claim."}
    frozen_test=PROJECT/"research/experiments/data/boundary_test"
    ledger={name:sha256(frozen_test/name) for name in ("freeze.json","freeze.sha256","test_access.json")}
    revision=PROJECT/"research/experiments/data/commands_v2/label_revisions"
    result={"created_utc":datetime.now(timezone.utc).isoformat(),"policy":"Audit only: no historical test rescoring.",
        "records":rows,"cohorts":cohorts,"evaluation_groups":groups,"malformed_records":failures,
        "cross_version_duplicate_groups":duplicates,"conflicting_canonical_label_groups":conflicts,
        "excluded_label_revision_archives":[str(p.resolve()) for p in revision.rglob("*.npz")],
        "historical_test_ledger_sha256":ledger,"fresh_v3_final_test_clips":0,
        "scope":"Numerical integrity/lineage/semantic compatibility, not independent visual label correctness or model generalization."}
    output=study/"dataset_audit.json"
    if output.exists():raise ValueError("Dataset audit already exists; preserve its snapshot.")
    write_json(output,result)
    if failures or conflicts:raise ValueError("Dataset integrity/label conflict needs resolution; audit saved.")
    print(json.dumps({"raw_files":len(rows),"unique_raw_recordings":len(identities),"malformed":len(failures),
        "label_conflicts":len(conflicts),"evaluation_groups":groups},indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("study",type=Path);audit(p.parse_args().study)


if __name__=="__main__":main()
