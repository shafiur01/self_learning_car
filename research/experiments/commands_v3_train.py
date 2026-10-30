"""Reproducible v3 supervised fitting; validation selects but never trains weights.

Reuses the existing features, eight-output temporal network and fixed fitter.
Saves every seed, with a strict new-contract loader. No camera activation/tests.
"""

import argparse
import copy
import json
import platform
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import boundary_train as fitting
from . import commands_v2 as v2
from . import commands_v3_collection as collection
from . import commands_v3_validation as validation
from .run import PROJECT, sha256, write_json

OUTPUT = PROJECT/"research/experiments/runs/commands_v3"
SOURCES = tuple(dict.fromkeys((*validation.SOURCES,"research/experiments/boundary_train.py",
    "research/experiments/run.py","research/experiments/commands_v3_train.py",
    "research/experiments/test_commands_v3_train.py","research/PROTOCOL.md","research/V3_VALIDATION_PLAN.md")))


def load_clip(path,contract):
    with np.load(path,allow_pickle=False) as a:
        m=json.loads(str(a["metadata"]))
        expected={"schema_version":3,"feature_version":1,"taxonomy":collection.TAXONOMY,
                  "labels":list(v2.LABELS),"convention":v2.CONVENTION,"command_contract":contract}
        if any(m.get(k)!=value for k,value in expected.items()) or m.get("split") not in ("train","validation") or m.get("label") not in v2.LABELS:
            raise ValueError("Recording is not compatible v3 development data.")
        xyz,times,visible=v2.validate_clip(a["xyz"],a["times"],a["visible"])
    if not np.isfinite(m.get("aspect",np.nan)) or m["aspect"]<=0:
        raise ValueError("Invalid recording aspect ratio.")
    return xyz,times,visible,m


def load_model(path):
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    contract=checkpoint.get("command_contract",{})
    try:expected_contract=collection.contract(contract["up_style"])
    except (KeyError,ValueError) as error:raise ValueError("Checkpoint lacks a known v3 gesture contract.") from error
    expected={"schema_version":3,"feature_version":1,"taxonomy":collection.TAXONOMY,
        "labels":list(v2.LABELS),"convention":v2.CONVENTION,"frames":v2.FRAMES,
        "features":v2.FEATURES,"seconds":v2.SECONDS,"command_contract":expected_contract}
    if any(checkpoint.get(k)!=value for k,value in expected.items()):
        raise ValueError("Checkpoint is not compatible with v3 meanings/features.")
    model=v2.build_model();model.load_state_dict(checkpoint["state_dict"]);model.eval()
    return model,checkpoint


def snapshot(root=collection.ROOT):
    root=Path(root);plan=validation.read_plan(root);state=validation.audit(root)
    if not state["complete"]:
        raise ValueError("Complete all48 validation clips before fitting; no run allocated.")
    roles={"train":[{**r,"split":"train"} for r in plan["training_records"]],
           "validation":[{**r,"split":"validation"} for r in state["records"]]}
    sessions={};identities=set();batches={};records=[]
    for role,rows in roles.items():
        batches[role]=[]
        for row in rows:
            if sha256(row["file"])!=row["sha256"]:raise ValueError("Pinned recording changed.")
            xyz,times,visible,m=load_clip(row["file"],plan["contract"])
            if any(m[k]!=row[k] for k in ("label","session","split")):
                raise ValueError("Manifest and recording roles/labels/session differ.")
            if m["session"] in sessions and sessions[m["session"]]!=role:
                raise ValueError("Session crosses training and validation.")
            sessions[m["session"]]=role
            identity=v2.content_sha(xyz,times,visible,m["aspect"])
            if identity in identities or identity!=row["raw_content_sha256"]:
                raise ValueError("Duplicate or changed raw recording identity.")
            identities.add(identity)
            batches[role].append((v2.features(xyz,times,visible,m["aspect"]),v2.LABELS.index(m["label"]),row["file"]))
            records.append({**row,"source":m.get("source"),"acquisition":m.get("acquisition"),
                            "up_shape":m.get("up_shape"),"passes_tracking_quality":v2.quality(times,visible)})
            if sha256(row["file"])!=row["sha256"]:raise ValueError("Recording changed while extracting features.")
        v2.require_classes(batches[role],role)
    manifest={"contract":plan["contract"],"records":records,
        "rows":{role:[r["file"] for r in rows] for role,rows in roles.items()},
        "counts":{role:dict(Counter(r["label"] for r in rows)) for role,rows in roles.items()},
        "quality_failures_retained":{role:sum(not r["passes_tracking_quality"] for r in records if r["split"]==role) for role in roles},
        "training_collection_plan_sha256":sha256(root/"plan.json"),
        "validation_collection_plan_sha256":sha256(root/"validation_plan.json"),
        "scope":"Supervised v3 single-user development, changed command meanings; no fixed-task v2 superiority/test/SSL/label-saving claim."}
    arrays={}
    for role,samples in batches.items():
        arrays[role+"_x"]=np.stack([s[0] for s in samples])
        arrays[role+"_y"]=np.array([s[1] for s in samples],dtype=np.int64)
    return plan,manifest,batches,arrays


def summarize(trials):
    result={"trials":trials,"metrics":{}}
    for field in fitting.FIELDS:
        values=[r["validation"][field] for r in trials]
        result["metrics"][field]={"mean":float(np.mean(values)),"run_sd":float(np.std(values,ddof=1)) if len(values)>1 else None}
    result["per_class"]={label:{"support_per_model":trials[0]["validation"]["per_class"][label]["support"],
        **{field:float(np.mean([r["validation"]["per_class"][label][field] for r in trials])) for field in ("precision","recall","f1")}}
        for label in v2.LABELS}
    result["interpretation"]="Checkpoint-selected validation of a changed gesture task; seeds share clips. No untouched test, population uncertainty, label savings, novel algorithm or robot reliability evidence."
    return result


def report(summary,manifest):
    lines=["# V3 supervised training and checkpoint-selected validation","",
        "All seeds retained. Validation selects weights; this is not an untouched test or a controlled comparison with the older gesture definition.","",
        "| Seed | Correct / validation clips | Accuracy | Macro F1 | NONE→motion | STOP recall | Saved update |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in summary["trials"]:
        m=r["validation"];correct=int(np.trace(m["confusion_matrix_actual_rows_predicted_columns"]))
        lines.append(f"| {r['seed']} | {correct}/{m['clips']} | {m['accuracy']:.2%} | {m['macro_f1']:.4f} | {m['no_command_to_motion_rate']:.2%} | {m['stop_recall']:.2%} | {r['best_step']} |")
    lines += ["","| Class | Training clips | Validation clips/model | Mean precision | Mean recall | Mean F1 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label,m in summary["per_class"].items():
        lines.append(f"| {label} | {manifest['counts']['train'][label]} | {m['support_per_model']} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.4f} |")
    lines += ["",f"Retained validation tracking failures: {manifest['quality_failures_retained']['validation']}.","",
        "No checkpoint activated in the camera. Keep all predictions/weights and inspect errors without automatically changing human labels."]
    return "\n".join(lines)+"\n"


def verify(output):
    output=Path(output);env=fitting.read_json(output/"environment.json");manifest=fitting.read_json(output/"manifest.json")
    config=fitting.read_json(output/"config.json");trials=fitting.read_json(output/"trials.json")
    if len(trials)!=len(config["seeds"]) or {r["seed"] for r in trials}!=set(config["seeds"]):
        raise ValueError("Require all declared seeds, without duplicates.")
    for name,digest in env["input_sha256"].items():
        if sha256(output/name)!=digest:raise ValueError("Archived development input changed.")
    for name,digest in env["source_sha256"].items():
        if sha256(output/"source"/name)!=digest:raise ValueError("Archived source changed.")
    for r in manifest["records"]:
        if sha256(r["file"])!=r["sha256"]:raise ValueError("Original recording changed.")
    with np.load(output/"development_features.npz",allow_pickle=False) as a:
        vx,vy=a["validation_x"].copy(),a["validation_y"].copy()
    hashes={};vectors=0
    for trial in trials:
        folder=output/f"seed_{trial['seed']}"
        if sha256(folder/"model.pt")!=trial["checkpoint_sha256"]:raise ValueError("Candidate changed.")
        model,checkpoint=load_model(folder/"model.pt")
        expected={"seed":trial["seed"],"command_contract":manifest["contract"],"manifest_sha256":sha256(output/"manifest.json"),
                  "training_files":manifest["rows"]["train"],"validation_files":manifest["rows"]["validation"],
                  "best_step":trial["best_step"],"development_files":[r["file"] for r in manifest["records"]],
                  "development_sessions":sorted({r["session"] for r in manifest["records"]}),
                  "validation_collection_plan_sha256":manifest["validation_collection_plan_sha256"]}
        if any(checkpoint.get(k)!=value for k,value in expected.items()):raise ValueError("Checkpoint provenance differs.")
        rows=fitting.read_json(folder/"validation_predictions.json")
        if [p["file"] for p in rows]!=manifest["rows"]["validation"] or [p["actual"] for p in rows]!=[v2.LABELS[int(i)] for i in vy]:
            raise ValueError("Prediction rows/labels differ from pinned inputs.")
        with torch.inference_mode():p=model(torch.from_numpy(vx)).softmax(1).numpy()
        np.testing.assert_allclose(p,[r["scores"] for r in rows],rtol=1e-6,atol=1e-7)
        if [r["predicted"] for r in rows]!=[v2.LABELS[int(i)] for i in p.argmax(1)]:raise ValueError("Saved predicted labels differ.")
        if fitting.metric_set(p,vy)!=trial["validation"] or fitting.read_json(folder/"metrics.json")!=trial:
            raise ValueError("Saved metrics do not reproduce from weights.")
        history=fitting.read_json(folder/"history.json")
        chosen=min(history,key=lambda h:h["selection_loss"])
        if (chosen["step"]!=trial["best_step"] or len(history)!=trial["validation_checks"]
                or trial["optimizer_steps"]!=config["optimizer_steps"]
                or trial["training_clips"]!=len(manifest["rows"]["train"])
                or not np.isclose(chosen["selection_loss"],trial["validation"]["nll"],rtol=1e-6,atol=1e-7)):
            raise ValueError("Checkpoint is not the declared minimum validation-loss update.")
        hashes[str((folder/"model.pt").relative_to(output))]=trial["checkpoint_sha256"];vectors+=len(rows)
    if fitting.read_json(output/"summary.json")!=summarize(trials):raise ValueError("Aggregated results differ.")
    return {"input_files_verified":len(manifest["records"]),"saved_prediction_vectors_verified":vectors,
            "checkpoints_sha256":hashes,"role_checkpoint_and_metric_provenance_verified":True,
            "scope":"Artifact/software integrity, not human label verification, generalization or robot reliability."}


def run(root=collection.ROOT,output_root=OUTPUT,config=None):
    root=Path(root);config=copy.deepcopy(v2.TRAIN_CONFIG if config is None else config)
    if config["augmentation"]!="none" or not config["seeds"] or len(set(config["seeds"]))!=len(config["seeds"]) or any(config[k]<=0 for k in ("optimizer_steps","validation_every_steps","batch_size","learning_rate","threads")):
        raise ValueError("Invalid fitting budget or augmentation.")
    torch.set_num_threads(config["threads"])
    plan,manifest,batches,arrays=snapshot(root)
    output=Path(output_root)/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True,exist_ok=False);trials=[]
    try:
        write_json(output/"status.json",{"state":"running","completed_trials":0})
        for name,value in (("config.json",config),("manifest.json",manifest),("contract.json",plan["contract"]),
            ("training_collection_plan.json",fitting.read_json(root/"plan.json")),("validation_collection_plan.json",plan)):
            write_json(output/name,value)
        np.savez_compressed(output/"development_features.npz",**arrays)
        source_hashes={}
        for name in SOURCES:
            destination=output/"source"/name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(PROJECT/name,destination);source_hashes[name]=sha256(destination)
        write_json(output/"environment.json",{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,
            "platform":platform.platform(),"source_sha256":source_hashes,"input_sha256":{name:sha256(output/name) for name in
                ("config.json","manifest.json","contract.json","training_collection_plan.json","validation_collection_plan.json","development_features.npz")}})
        attempts={role:[fitting.read_json(p) for p in sorted((root/folder).glob("*.json"))]
                  for role,folder in (("up_training","recording_attempts"),("validation","validation_attempts"))}
        write_json(output/"costs.json",{"reused_training_labels":len(manifest["rows"]["train"])-50,
            "new_up_training_labels":50,"new_validation_labels":48,"attempts":attempts,
            "logged_camera_seconds":{role:sum(a["wall_seconds"] for a in rows) for role,rows in attempts.items()},
            "scope":"Camera/replay/preparation/pause intervals, not pure labeling; historical collection/corrections/upstream pretraining not zero-cost."})
        print(f"Study folder: {output}",flush=True)
        for seed in config["seeds"]:
            folder=output/f"seed_{seed}";folder.mkdir()
            model,history,stats=fitting.fit_candidate(arrays["train_x"],arrays["train_y"],arrays["validation_x"],arrays["validation_y"],seed,config)
            torch.save({"state_dict":model.state_dict(),"schema_version":3,"feature_version":1,"taxonomy":collection.TAXONOMY,
                "labels":list(v2.LABELS),"convention":v2.CONVENTION,"frames":v2.FRAMES,"features":v2.FEATURES,"seconds":v2.SECONDS,
                "command_contract":plan["contract"],"seed":seed,"best_step":stats["best_step"],"manifest_sha256":sha256(output/"manifest.json"),
                "training_files":manifest["rows"]["train"],"validation_files":manifest["rows"]["validation"],
                "development_files":[r["file"] for r in manifest["records"]],"development_sessions":sorted({r["session"] for r in manifest["records"]}),
                "validation_collection_plan_sha256":manifest["validation_collection_plan_sha256"]},folder/"model.pt")
            write_json(folder/"history.json",history)
            model,_=load_model(folder/"model.pt")
            with torch.inference_mode():p=model(torch.from_numpy(arrays["validation_x"])).softmax(1).numpy()
            trial={**stats,"checkpoint_sha256":sha256(folder/"model.pt"),"validation":fitting.metric_set(p,arrays["validation_y"])}
            write_json(folder/"validation_predictions.json",v2.predictions(batches["validation"],p))
            write_json(folder/"metrics.json",trial);trials.append(trial)
            write_json(output/"trials.json",trials);write_json(output/"status.json",{"state":"running","completed_trials":len(trials)})
            print(f"seed={seed}: validation accuracy={trial['validation']['accuracy']:.2%}, F1={trial['validation']['macro_f1']:.4f}, saved step={trial['best_step']}",flush=True)
        summary=summarize(trials);write_json(output/"summary.json",summary)
        (output/"report.md").write_text(report(summary,manifest))
        checked=verify(output);validation.read_plan(root)
        write_json(output/"verification.json",checked);write_json(output/"status.json",{"state":"complete","completed_trials":len(trials)})
    except BaseException as error:
        write_json(output/"status.json",{"state":"failed","completed_trials":len(trials),"error":str(error)})
        raise
    print(f"All seeds saved and verified: {output/'report.md'}. Camera application unchanged.",flush=True)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="action",required=True)
    p=sub.add_parser("train");p.add_argument("--root",type=Path,default=collection.ROOT)
    p=sub.add_parser("verify");p.add_argument("run",type=Path)
    args=parser.parse_args()
    if args.action=="train":run(args.root)
    else:print(verify(args.run))


if __name__=="__main__":main()
