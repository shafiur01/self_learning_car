"""Fixed-weight validation sensitivity probes; no fitting or new test claim.

Modified inputs need not retain gesture meaning. Metrics for these inputs are
agreement with original human labels, not accuracy on valid newly labeled data.
"""

import argparse
import shutil
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch

from . import commands_v3_train as training
from . import commands_v2 as v2
from .boundary_train import metric_set,read_json,state_hash
from .run import PROJECT,sha256,write_json

STUDY=training.OUTPUT/"20261012T205727638067Z"
OUTPUT=PROJECT/"research/experiments/runs/motion_probes"
VARIANTS=("baseline","zero_wrist_xy","zero_size_change","static_local_shape",
          "static_shape_zero_trajectory","shuffled_frames","reversed_raw")
SHUFFLE_SEED=20261012


def perturb(x,variant,permutation=None):
    if x.ndim!=3 or x.shape[1:]!=(32,67):raise ValueError("Expected N×32×67 feature inputs.")
    result=x.copy()
    if variant=="zero_wrist_xy":result[:,:,63:65]=0
    elif variant=="zero_size_change":result[:,:,65]=0
    elif variant in ("static_local_shape","static_shape_zero_trajectory"):
        result[:,:,:63]=result[:,16:17,:63]
        if variant=="static_shape_zero_trajectory":result[:,:,63:66]=0
    elif variant=="shuffled_frames":
        if permutation is None or sorted(np.asarray(permutation).tolist())!=list(range(32)):
            raise ValueError("Require a declared full permutation.")
        result=result[:,permutation,:].copy()
    elif variant!="baseline":raise ValueError("Raw reversal requires coordinate re-extraction.")
    return result


def reverse_features(xyz,times,visible,aspect):
    # Reverse physical coordinate samples, then restore increasing clock values;
    # the original feature extractor reanchors wrist/size at the new beginning.
    return v2.features(xyz[::-1].copy(),times[-1]-times[::-1],visible[::-1].copy(),aspect)


def js_divergence(p,q):
    p=np.maximum(p,1e-12);q=np.maximum(q,1e-12);mid=(p+q)/2
    return float(.5*np.mean(np.sum(p*np.log(p/mid)+q*np.log(q/mid),axis=1)))


def summarize(trials,seeds):
    by={(r["variant"],r["seed"]):r for r in trials}
    if len(by)!=len(trials) or set(by)!={(v,s) for v in VARIANTS for s in seeds}:
        raise ValueError("Require every fixed variant/seed without duplicates.")
    result={}
    for variant in VARIANTS:
        rows=[by[variant,s] for s in seeds]
        result[variant]={"original_label_accuracy_mean":float(np.mean([r["original_label_metrics"]["accuracy"] for r in rows])),
            "original_label_macro_f1_mean":float(np.mean([r["original_label_metrics"]["macro_f1"] for r in rows])),
            "prediction_flip_rate_mean":float(np.mean([r["prediction_flip_rate"] for r in rows])),
            "mean_js_divergence":float(np.mean([r["js_divergence"] for r in rows])),
            "original_label_recall":{label:float(np.mean([r["original_label_metrics"]["per_class"][label]["recall"] for r in rows])) for label in v2.LABELS}}
    return {"variants":result,"interpretation":"Exploratory fixed-weight sensitivity on inspected development validation. Perturbed clips may be OOD/change meaning; no new accuracy, causality, novelty, label-savings or robot-reliability claim."}


def run(study=STUDY,output_root=OUTPUT):
    study=Path(study);torch.set_num_threads(2)
    config=read_json(study/"config.json");seeds=config["seeds"]
    if read_json(study/"status.json")!={"state":"complete","completed_trials":len(seeds)}:
        raise ValueError("Require the completed fixed v3 study.")
    verification=training.verify(study)
    manifest=read_json(study/"manifest.json");by_file={r["file"]:r for r in manifest["records"]}
    rows=[by_file[f] for f in manifest["rows"]["validation"]]
    if any(r["split"]!="validation" for r in rows):raise ValueError("Only validation rows allowed.")
    with np.load(study/"development_features.npz",allow_pickle=False) as a:
        x,y=a["validation_x"].copy(),a["validation_y"].copy()
    permutation=np.random.default_rng(SHUFFLE_SEED).permutation(32)
    arrays={v:perturb(x,v,permutation) for v in VARIANTS if v!="reversed_raw"}
    reversed_rows=[]
    for row in rows:
        xyz,times,visible,m=training.load_clip(row["file"],manifest["contract"])
        if m["split"]!="validation":raise ValueError("Refuse a non-validation payload.")
        np.testing.assert_array_equal(v2.features(xyz,times,visible,m["aspect"]),x[len(reversed_rows)])
        reversed_rows.append(reverse_features(xyz,times,visible,m["aspect"]))
    arrays["reversed_raw"]=np.stack(reversed_rows)
    output=Path(output_root)/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True,exist_ok=False);trials=[]
    try:
        write_json(output/"status.json",{"state":"running","completed_predictions":0})
        write_json(output/"config.json",{"study":str(study.resolve()),"seeds":seeds,"variants":list(VARIANTS),
            "shuffle_seed":SHUFFLE_SEED,"frame_permutation":permutation.tolist(),"center_sample":16,
            "checkpoint_sha256":verification["checkpoints_sha256"],"visibility":"retained except follows frames for shuffle/reversal",
            "scope":"Fixed weights; original-label agreement only for counterfactual inputs; no test data."})
        write_json(output/"manifest.json",{"records":rows,"original_manifest_sha256":sha256(study/"manifest.json"),
            "original_features_sha256":sha256(study/"development_features.npz"),"contract":manifest["contract"]})
        np.savez_compressed(output/"probe_features.npz",y=y,**arrays)
        for name in ("research/experiments/motion_probe.py","research/experiments/test_motion_probe.py",
                     "gesture_commands/data.py","gesture_commands/model.py","research/experiments/commands_v3_train.py",
                     "research/PROTOCOL.md"):
            destination=output/"source"/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/name,destination)
        samples=[(x[i],int(y[i]),r["file"]) for i,r in enumerate(rows)]
        for seed in seeds:
            model,_=training.load_model(study/f"seed_{seed}/model.pt");initial_state=state_hash(model)
            with torch.inference_mode():base=model(torch.from_numpy(x)).softmax(1).numpy()
            saved=read_json(study/f"seed_{seed}/validation_predictions.json")
            np.testing.assert_allclose(base,[r["scores"] for r in saved],rtol=1e-6,atol=1e-7)
            for variant in VARIANTS:
                with torch.inference_mode():p=model(torch.from_numpy(arrays[variant])).softmax(1).numpy()
                result={"seed":seed,"variant":variant,"original_label_metrics":metric_set(p,y),
                    "prediction_flip_rate":float(np.mean(p.argmax(1)!=base.argmax(1))),"js_divergence":js_divergence(base,p)}
                folder=output/variant/f"seed_{seed}";folder.mkdir(parents=True)
                write_json(folder/"predictions.json",v2.predictions(samples,p));write_json(folder/"metrics.json",result)
                trials.append(result)
            if state_hash(model)!=initial_state:raise ValueError("Weights changed during probe.")
            if sha256(study/f"seed_{seed}/model.pt")!=verification["checkpoints_sha256"][f"seed_{seed}/model.pt"]:
                raise ValueError("Saved checkpoint changed during probe.")
        for row in rows:
            if sha256(row["file"])!=row["sha256"]:raise ValueError("Validation recording changed.")
        write_json(output/"trials.json",trials);summary=summarize(trials,seeds);write_json(output/"summary.json",summary)
        lines=["# Fixed-weight motion sensitivity on inspected v3 validation","",
            "Perturbed inputs may change the gesture meaning or leave the data distribution. Agreement with original labels below is not a newly valid accuracy benchmark.","",
            "| Probe | Original-label agreement | Prediction changes vs baseline | Macro F1 vs original labels |",
            "| --- | ---: | ---: | ---: |"]
        for v,r in summary["variants"].items():lines.append(f"| {v} | {r['original_label_accuracy_mean']:.2%} | {r['prediction_flip_rate_mean']:.2%} | {r['original_label_macro_f1_mean']:.4f} |")
        (output/"report.md").write_text("\n".join(lines)+"\n")
        write_json(output/"verification.json",{"unchanged_checkpoints":verification["checkpoints_sha256"],
            "baseline_vectors_reproduced":len(rows)*len(seeds),"probe_vectors_saved":len(rows)*len(trials),
            "no_weight_updates":True,"no_test_payloads":True})
        write_json(output/"artifacts_sha256.json",{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name!="status.json"})
        write_json(output/"status.json",{"state":"complete","completed_predictions":len(trials)*len(rows)})
    except BaseException as error:
        write_json(output/"status.json",{"state":"failed","completed_predictions":len(trials)*len(rows),"error":str(error)});raise
    print(output);print({k:round(v["prediction_flip_rate_mean"],4) for k,v in summary["variants"].items()})
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--study",type=Path,default=STUDY)
    args=parser.parse_args();run(args.study)
