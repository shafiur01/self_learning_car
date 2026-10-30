"""Experiment011: motion features across model families and fixed controls."""

import argparse
import copy
import hashlib
import platform
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import motion_representation as motion
from .run import PROJECT, sha256, write_json

SOURCE = motion.OUTPUT / "20261015T051753816210Z"
OUTPUT = PROJECT / "research/experiments/runs/cross_model_motion"
FAMILIES = ("cnn", "lstm", "bilstm", "local_set", "feature_set")
COUNTS = {"cnn":49484,"lstm":49276,"bilstm":49656,"local_set":49163,"feature_set":49163,"causal_cnn":49632}
SOURCES = tuple(dict.fromkeys((*motion.SOURCES,"research/experiments/cross_model_motion.py",
    "research/experiments/test_cross_model_motion.py","research/experiments/dataset_validation.py",
    "research/CROSS_MODEL_MOTION_PROTOCOL.md")))


class SetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(146,128),nn.ReLU(),nn.Linear(128,131),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(131,96),nn.ReLU(),nn.Dropout(.2),nn.Linear(96,8))

    def forward(self,x):
        return self.head(self.encoder(x).mean(1))


class CNN(nn.Module):
    def __init__(self,causal=False):
        super().__init__()
        if causal:
            layers=[]
            for cin,cout,dilation in ((146,40,1),(40,48,2),(48,48,4)):
                layers.extend((nn.ConstantPad1d((2*dilation,0),0),nn.Conv1d(cin,cout,3,dilation=dilation),nn.ReLU()))
            self.features=nn.Sequential(*layers)
            hidden=96
        else:
            self.features=nn.Sequential(nn.Conv1d(146,36,5,padding=2),nn.ReLU(),nn.Conv1d(36,48,5,padding=2),nn.ReLU())
            hidden=72
        self.head=nn.Sequential(nn.AdaptiveAvgPool1d(4),nn.Flatten(),nn.Linear(192,hidden),nn.ReLU(),nn.Dropout(.2),nn.Linear(hidden,8))

    def forward(self,x):
        return self.head(self.features(x.transpose(1,2)))


class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent=nn.LSTM(146,25,batch_first=True,bidirectional=True)
        self.head=nn.Sequential(nn.AdaptiveAvgPool1d(4),nn.Flatten(),nn.Linear(200,72),nn.ReLU(),nn.Dropout(.2),nn.Linear(72,8))

    def forward(self,x):
        sequence,_=self.recurrent(x)
        return self.head(sequence.transpose(1,2))


def build_model(family):
    if family=="lstm":model=motion.Model()
    elif family=="bilstm":model=BiLSTM()
    elif family in ("local_set","feature_set"):model=SetModel()
    elif family in ("cnn","causal_cnn"):model=CNN(family=="causal_cnn")
    else:raise ValueError("Unknown model family.")
    if sum(p.numel() for p in model.parameters())!=COUNTS[family]:
        raise ValueError("Undeclared parameter-count change.")
    return model


def inputs(x,family,variant="standard"):
    if x.ndim!=3 or x.shape[1:]!=(32,146):raise ValueError("Expected N×32×146 inputs.")
    if variant not in ("standard","no_elapsed"):raise ValueError("Unknown input control.")
    if family=="local_set" or variant=="no_elapsed":
        x=x.copy()
        if family=="local_set":x[:,:,63:66]=0;x[:,:,130:133]=0
        if variant=="no_elapsed":x[:,:,144]=0
    return x


def fit(family,arrays,seed,config):
    if family=="lstm":return motion.fit(arrays,seed,config)
    tx,ty=torch.from_numpy(arrays["train_x"]),torch.from_numpy(arrays["train_y"])
    vx,vy=torch.from_numpy(arrays["validation_x"]),torch.from_numpy(arrays["validation_y"])
    counts=np.bincount(arrays["train_y"],minlength=8)
    if np.any(counts<5):raise ValueError("Need five training samples per class.")
    torch.manual_seed(seed);model=build_model(family)
    initial=motion.previous.fitting.state_hash(model)
    rng=torch.Generator().manual_seed(seed+1_000_000)
    optimizer=torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(counts.sum()/(8*counts),dtype=torch.float32))
    best_loss,best_state,best_step=float("inf"),None,None
    history,batch_hash=[],hashlib.sha256();started=time.perf_counter()
    for step in range(1,config["optimizer_steps"]+1):
        model.train();indices=torch.randint(len(tx),(config["batch_size"],),generator=rng)
        batch_hash.update(indices.numpy().tobytes());optimizer.zero_grad()
        loss=loss_fn(model(tx[indices]),ty[indices])
        if not torch.isfinite(loss):raise ValueError("Nonfinite training loss.")
        loss.backward();optimizer.step()
        if step%config["validation_every_steps"]==0 or step==config["optimizer_steps"]:
            model.eval()
            with torch.inference_mode():value=nn.functional.cross_entropy(model(vx),vy).item()
            if not np.isfinite(value):raise ValueError("Nonfinite validation loss.")
            history.append({"step":step,"selection_loss":value})
            if value<best_loss:best_loss,best_state,best_step=value,copy.deepcopy(model.state_dict()),step
    model.load_state_dict(best_state);model.eval()
    return model,history,{"seed":seed,"best_step":best_step,"training_seconds":time.perf_counter()-started,
        "initial_state_sha256":initial,"batch_indices_sha256":batch_hash.hexdigest(),"optimizer_steps":config["optimizer_steps"],
        "training_class_counts":counts.tolist(),"validation_checks":len(history)}


def jobs(config,stage):
    seeds=config["training"]["seeds"]
    if stage=="core":return [(family,mode,"standard",seed) for family in FAMILIES for mode in motion.rep.MODES for seed in seeds]
    if stage=="controls":return [(family,"camera_motion",variant,seed) for family,variant in
        (("causal_cnn","standard"),("feature_set","no_elapsed"),("lstm","no_elapsed")) for seed in seeds]
    raise ValueError("Unknown phase.")


def folder_name(family,mode,variant,seed):
    return f"{family}_{mode}_{variant}_seed_{seed}"


def load_model(path):
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    expected={"experiment_id":"011_cross_model_motion","schema_version":3,"feature_version":2,
        "features":146,"frames":32,"labels":list(motion.previous.v2.LABELS)}
    contract=checkpoint.get("command_contract",{})
    try:expected["command_contract"]=motion.previous.training.collection.contract(contract["up_style"])
    except (KeyError,ValueError) as e:raise ValueError("Missing gesture contract.") from e
    if (any(checkpoint.get(k)!=v for k,v in expected.items()) or checkpoint.get("mode") not in motion.rep.MODES
        or checkpoint.get("variant") not in ("standard","no_elapsed")):
        raise ValueError("Incompatible cross-model checkpoint.")
    model=build_model(checkpoint["family"]);model.load_state_dict(checkpoint["state_dict"]);model.eval()
    return model,checkpoint


def prepare(source=SOURCE,output_root=OUTPUT):
    source=Path(source);checked=motion.verify(source)
    config=motion.previous.fitting.read_json(source/"config.json")
    config={**config,"experiment_id":"011_cross_model_motion","source_study":str(source.resolve()),
        "families":list(FAMILIES),"parameter_counts":COUNTS,"reference_checkpoints":checked["checkpoints_sha256"],
        "controls":[["causal_cnn","standard"],["feature_set","no_elapsed"],["lstm","no_elapsed"]]}
    output=Path(output_root)/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"config.json",config)
    for name in ("manifest.json","features.npz","feature_audits.json","prefix_observations.json"):
        shutil.copyfile(source/name,output/name)
    hashes={}
    for name in SOURCES:
        target=output/"source"/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/name,target)
        hashes[name]=sha256(target)
    write_json(output/"environment.json",{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,
        "platform":platform.platform(),"source_sha256":hashes,"input_sha256":{name:sha256(output/name) for name in
        ("config.json","manifest.json","features.npz","feature_audits.json","prefix_observations.json")}})
    write_json(output/"trials.json",[]);write_json(output/"status.json",{"state":"prepared","completed_trials":0})
    print(f"Prepared study: {output}",flush=True)
    return output


def summarize(trials,config):
    result={"groups":{},"camera_motion_minus_pose":{},"controls":{}}
    for family in FAMILIES:
        for mode in motion.rep.MODES:
            rows=[r for r in trials if r["family"]==family and r["mode"]==mode and r["variant"]=="standard"]
            if not rows:continue
            result["groups"][f"{family}/{mode}"]={"family":family,"mode":mode,"parameters":COUNTS[family],
                "metrics":{field:{"mean":float(np.mean([r["validation"][field] for r in rows])),
                    "seed_sd":float(np.std([r["validation"][field] for r in rows],ddof=1)) if len(rows)>1 else None}
                    for field in motion.previous.fitting.FIELDS},
                "class_recall":{label:float(np.mean([r["validation"]["per_class"][label]["recall"] for r in rows])) for label in motion.previous.v2.LABELS},
                "prefix_agreement":{tag:float(np.mean([r["prefix_intent_metrics"][tag]["accuracy"] for r in rows])) for tag in motion.DEADLINES},
                "training_seconds_mean":float(np.mean([r["training_seconds"] for r in rows])),
                "latency_ms_mean":float(np.mean([r["latency"]["median_ms"] for r in rows]))}
        diffs=[]
        for seed in config["training"]["seeds"]:
            rows={r["mode"]:r for r in trials if r["family"]==family and r["variant"]=="standard" and r["seed"]==seed}
            if "camera_motion" in rows and "camera_pose" in rows:
                diffs.append(rows["camera_motion"]["validation"]["macro_f1"]-rows["camera_pose"]["validation"]["macro_f1"])
        if diffs:result["camera_motion_minus_pose"][family]={"differences":diffs,"mean":float(np.mean(diffs))}
    for family,variant in config["controls"]:
        rows=[r for r in trials if r["family"]==family and r["variant"]==variant and r["stage"]=="controls"]
        if rows:result["controls"][family+"/"+variant]={"accuracy":float(np.mean([r["validation"]["accuracy"] for r in rows])),
            "macro_f1":float(np.mean([r["validation"]["macro_f1"] for r in rows])),
            "none_to_motion":float(np.mean([r["validation"]["no_command_to_motion_rate"] for r in rows])),
            "stop_recall":float(np.mean([r["validation"]["stop_recall"] for r in rows])),
            "prefix_agreement":{tag:float(np.mean([r["prefix_intent_metrics"][tag]["accuracy"] for r in rows])) for tag in motion.DEADLINES}}
    result["scope"]="Inspected checkpoint-selected single-user development; no untouched-test, causal-memory, SOTA, label-saving or robot claim."
    return result


def train(output,stage="core"):
    output=Path(output);config=motion.previous.fitting.read_json(output/"config.json")
    manifest=motion.previous.fitting.read_json(output/"manifest.json");env=motion.previous.fitting.read_json(output/"environment.json")
    for name,digest in env["input_sha256"].items():
        if sha256(output/name)!=digest:raise ValueError("Pinned study inputs changed.")
    for name,digest in env["source_sha256"].items():
        if name.endswith(".py") and sha256(PROJECT/name)!=digest:raise ValueError("Executing sources differ from archive.")
    torch.set_num_threads(config["training"]["threads"])
    trials=motion.previous.fitting.read_json(output/"trials.json");completed={(r["family"],r["mode"],r["variant"],r["seed"]) for r in trials}
    try:
        with np.load(output/"features.npz",allow_pickle=False) as data:
            for family,mode,variant,seed in jobs(config,stage):
                if (family,mode,variant,seed) in completed:continue
                folder=output/folder_name(family,mode,variant,seed);folder.mkdir(exist_ok=False)
                print(f"Fitting {family}/{mode}/{variant}, seed {seed}...",flush=True)
                arrays={"train_x":inputs(data[mode+"_train_x"],family,variant),"validation_x":inputs(data[mode+"_validation_x"],family,variant),
                    "train_y":data["train_y"],"validation_y":data["validation_y"]}
                reference=family=="lstm" and variant=="standard"
                if reference:
                    origin=Path(config["source_study"])/f"{mode}_seed_{seed}"
                    model,_=motion.load_model(origin/"model.pt")
                    prior=motion.previous.fitting.read_json(origin/"metrics.json")
                    stats={k:v for k,v in prior.items() if k not in ("mode","validation","prefix_intent_metrics","checkpoint_sha256","latency")}
                    history=motion.previous.fitting.read_json(origin/"history.json")
                    shutil.copyfile(origin/"model.pt",folder/"original_reference.pt")
                else:model,history,stats=fit(family,arrays,seed,config["training"])
                torch.save({"state_dict":model.state_dict(),"experiment_id":config["experiment_id"],"schema_version":3,"feature_version":2,
                    "features":146,"frames":32,"labels":list(motion.previous.v2.LABELS),"command_contract":manifest["contract"],
                    "family":family,"mode":mode,"variant":variant,"seed":seed,"best_step":stats["best_step"],
                    "config_sha256":sha256(output/"config.json"),"manifest_sha256":sha256(output/"manifest.json"),
                    "training_files":manifest["rows"]["train"],"validation_files":manifest["rows"]["validation"]},folder/"model.pt")
                write_json(folder/"history.json",history);model,_=load_model(folder/"model.pt")
                metrics={}
                for tag in ("full",*motion.DEADLINES):
                    x=inputs(data[f"{mode}_validation_x" if tag=="full" else f"{mode}_{tag}_x"],family,variant)
                    with torch.inference_mode():p=model(torch.from_numpy(x)).softmax(1).numpy()
                    samples=list(zip(x,data["validation_y"],manifest["rows"]["validation"]))
                    write_json(folder/f"{tag}_predictions.json",motion.previous.v2.predictions(samples,p))
                    metrics[tag]=motion.previous.fitting.metric_set(p,data["validation_y"])
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                trial={**stats,"family":family,"mode":mode,"variant":variant,"stage":stage,"reference_reused":reference,
                    "additional_training_seconds":0.0 if reference else stats["training_seconds"],"parameters":COUNTS[family],
                    "checkpoint_sha256":sha256(folder/"model.pt"),"state_sha256":motion.previous.fitting.state_hash(model),
                    "validation":metrics.pop("full"),"prefix_intent_metrics":metrics,
                    "latency":motion.previous.window_latency(model,torch.from_numpy(arrays["validation_x"][:1]),40,200),
                    "cumulative_process_peak_rss_bytes":int(rss if sys.platform=="darwin" else rss*1024),
                    "memory_scope":"Cumulative process high-water RSS, not isolated per-model memory; not comparable across model order."}
                if reference and trial["validation"]!=prior["validation"]:raise ValueError("Reference predictions changed.")
                write_json(folder/"metrics.json",trial);trials.append(trial);write_json(output/"trials.json",trials)
                write_json(output/"status.json",{"state":stage+"_running","completed_trials":len(trials)})
                print(f"{family}/{mode}/{variant} seed={seed}: accuracy={trial['validation']['accuracy']:.2%}, F1={trial['validation']['macro_f1']:.4f}",flush=True)
        write_json(output/"summary.json",summarize(trials,config))
        write_json(output/"verification.json",verify(output))
        write_json(output/"status.json",{"state":stage+"_complete","completed_trials":len(trials)})
    except BaseException as e:
        write_json(output/"status.json",{"state":"failed","completed_trials":len(trials),"error":str(e)})
        raise
    return output


def verify(output):
    output=Path(output);config=motion.previous.fitting.read_json(output/"config.json")
    manifest=motion.previous.fitting.read_json(output/"manifest.json");env=motion.previous.fitting.read_json(output/"environment.json")
    trials=motion.previous.fitting.read_json(output/"trials.json");keys=set();states={};streams={};vectors=0;hashes={}
    for name,digest in env["input_sha256"].items():
        if sha256(output/name)!=digest:raise ValueError("Archived input changed.")
    for name,digest in env["source_sha256"].items():
        if sha256(output/"source"/name)!=digest:raise ValueError("Archived source changed.")
    for row in manifest["records"]:
        if sha256(row["file"])!=row["sha256"]:raise ValueError("Original recording changed.")
    torch.set_num_threads(config["training"]["threads"])
    with np.load(output/"features.npz",allow_pickle=False) as data:
        for trial in trials:
            f,m,v,s=(trial[k] for k in ("family","mode","variant","seed"));key=(f,m,v,s)
            if key in keys:raise ValueError("Duplicate trial.")
            keys.add(key);folder=output/folder_name(*key)
            if sha256(folder/"model.pt")!=trial["checkpoint_sha256"]:raise ValueError("Weight changed.")
            model,c=load_model(folder/"model.pt")
            expected={"family":f,"mode":m,"variant":v,"seed":s,"best_step":trial["best_step"],"command_contract":manifest["contract"],
                "config_sha256":sha256(output/"config.json"),"manifest_sha256":sha256(output/"manifest.json"),
                "training_files":manifest["rows"]["train"],"validation_files":manifest["rows"]["validation"]}
            if any(c.get(k)!=value for k,value in expected.items()):raise ValueError("Checkpoint provenance differs.")
            for tag in ("full",*motion.DEADLINES):
                x=inputs(data[f"{m}_validation_x" if tag=="full" else f"{m}_{tag}_x"],f,v)
                with torch.inference_mode():p=model(torch.from_numpy(x)).softmax(1).numpy()
                rows=motion.previous.fitting.read_json(folder/f"{tag}_predictions.json")
                if [r["file"] for r in rows]!=manifest["rows"]["validation"] or [r["actual"] for r in rows]!=[motion.previous.v2.LABELS[int(y)] for y in data["validation_y"]]:
                    raise ValueError("Prediction rows/labels changed.")
                np.testing.assert_allclose(p,[r["scores"] for r in rows],rtol=1e-6,atol=1e-7)
                if [r["predicted"] for r in rows]!=[motion.previous.v2.LABELS[int(i)] for i in p.argmax(1)]:raise ValueError("Predicted names changed.")
                if motion.previous.fitting.metric_set(p,data["validation_y"])!=(trial["validation"] if tag=="full" else trial["prefix_intent_metrics"][tag]):
                    raise ValueError("Prediction metrics differ.")
                vectors+=len(rows)
            history=motion.previous.fitting.read_json(folder/"history.json");chosen=min(history,key=lambda r:r["selection_loss"])
            steps=list(range(config["training"]["validation_every_steps"],config["training"]["optimizer_steps"]+1,config["training"]["validation_every_steps"]))
            if not steps or steps[-1]!=config["training"]["optimizer_steps"]:steps.append(config["training"]["optimizer_steps"])
            if (chosen["step"]!=trial["best_step"] or len(history)!=trial["validation_checks"]
                or [r["step"] for r in history]!=steps or trial["parameters"]!=COUNTS[f]
                or trial["training_class_counts"]!=[manifest["counts"]["train"][label] for label in motion.previous.v2.LABELS]
                or trial["optimizer_steps"]!=config["training"]["optimizer_steps"]
                or not np.isclose(chosen["selection_loss"],trial["validation"]["nll"],rtol=1e-6,atol=1e-7)
                or motion.previous.fitting.read_json(folder/"metrics.json")!=trial):raise ValueError("Selection/metrics provenance differs.")
            if motion.previous.fitting.state_hash(model)!=trial["state_sha256"]:raise ValueError("Model state differs.")
            states.setdefault((f,s),set()).add(trial["initial_state_sha256"]);streams.setdefault(s,set()).add(trial["batch_indices_sha256"])
            hashes[str((folder/"model.pt").relative_to(output))]=trial["checkpoint_sha256"]
            if trial["reference_reused"]:
                source=Path(config["source_study"])/f"{m}_seed_{s}/model.pt"
                if sha256(source)!=config["reference_checkpoints"][str(source.relative_to(Path(config["source_study"])))] or sha256(folder/"original_reference.pt")!=sha256(source):
                    raise ValueError("Saved reference changed.")
                original,_=motion.load_model(source)
                if motion.previous.fitting.state_hash(original)!=trial["state_sha256"]:raise ValueError("Reference state differs.")
    present_core={k for k in keys if k[0] in FAMILIES and k[2]=="standard"}
    if present_core!=set(jobs(config,"core")):raise ValueError("Incomplete core comparison.")
    controls={k for k in keys if k not in present_core}
    if controls and controls!=set(jobs(config,"controls")):raise ValueError("Incomplete control phase.")
    if any(len(value)!=1 for value in (*states.values(),*streams.values())):raise ValueError("Unpaired initialization or sample streams.")
    if motion.previous.fitting.read_json(output/"summary.json")!=summarize(trials,config):raise ValueError("Summary differs.")
    return {"input_files_verified":len(manifest["records"]),"checkpoints_verified":len(trials),"prediction_vectors_verified":vectors,
        "sampling_and_within_family_initialization_paired":True,"checkpoints_sha256":hashes}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    sub.add_parser("prepare")
    for name in ("core","controls","verify"):sub.add_parser(name).add_argument("study",type=Path)
    args=p.parse_args()
    if args.action=="prepare":prepare()
    elif args.action=="verify":print(verify(args.study))
    else:train(args.study,args.action)


if __name__=="__main__":main()
