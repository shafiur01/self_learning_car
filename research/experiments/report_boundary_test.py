"""Reproduce archived test scores and derive reports, without reopening raw clips.

Post-evaluation artifact verification/description, not another test invocation or
model selection. Preserves the frozen collector/evaluator and original outputs.
"""

import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import boundary_test as test
from . import boundary_train as training
from . import commands_v2 as v2
from .run import PROJECT, sha256, write_json


def run(root=test.ROOT):
    root = Path(root)
    plan = test.read_freeze(root)
    output = root/"evaluation"
    for name, digest in training.read_json(output/"artifacts_sha256.json").items():
        if sha256(output/name) != digest:
            raise ValueError("Original evaluation artifact changed.")
    ledger = training.read_json(root/"test_access.json")
    if len(ledger["accesses"]) != 1 or ledger["accesses"][0]["state"] != "complete":
        raise ValueError("Require one completed joint test access.")
    manifest = training.read_json(output/"manifest.json")
    records = manifest["records"]
    if manifest["freeze_sha256"] != sha256(root/"freeze.json") or len(records) != 80:
        raise ValueError("Test manifest differs from declared pool.")
    for record in records:
        if sha256(record["file"]) != record["sha256"]:
            raise ValueError("Test recording bytes changed after evaluation.")
    with np.load(output/"features.npz", allow_pickle=False) as a:
        x, y = a["x"].copy(), a["y"].copy()
    if [v2.LABELS[int(i)] for i in y] != [r["label"] for r in records]:
        raise ValueError("Archived labels/manifest row alignment differs.")
    trials = training.read_json(output/"trials.json")
    by_key = {(r["condition"],r["seed"]):r for r in trials}
    expected = {(r["condition"],r["seed"]) for r in plan["checkpoints"]}
    if set(by_key) != expected or len(trials) != len(expected):
        raise ValueError("Require all frozen candidates, exactly once.")
    torch.set_num_threads(2)
    diagnostics = {}
    vectors = 0
    for candidate in plan["checkpoints"]:
        condition, seed = candidate["condition"], candidate["seed"]
        folder = output/condition/f"seed_{seed}"
        guesses = training.read_json(folder/"predictions.json")
        result = by_key[condition,seed]
        if training.read_json(folder/"metrics.json") != result or result["checkpoint_sha256"] != candidate["sha256"]:
            raise ValueError("Candidate report differs from joint trial or frozen checkpoint.")
        if [(p["file"],p["actual"]) for p in guesses] != [(r["file"],r["label"]) for r in records]:
            raise ValueError("Prediction rows or human labels differ from test manifest.")
        model,_ = v2.load_model(Path(plan["study"])/candidate["file"])
        with torch.inference_mode():
            p = model(torch.from_numpy(x)).softmax(1).numpy()
        np.testing.assert_allclose(p,[r["scores"] for r in guesses],rtol=1e-6,atol=1e-7)
        if [v2.LABELS[int(i)] for i in p.argmax(1)] != [g["predicted"] for g in guesses]:
            raise ValueError("Saved class names disagree with saved/model scores.")
        if training.metric_set(p,y) != result["pooled"]:
            raise ValueError("Pooled metrics do not reproduce.")
        for sitting in (1,2):
            indices = [i for i,r in enumerate(records) if r["sitting"] == sitting]
            if len(indices) != 40 or Counter(records[i]["label"] for i in indices) != Counter(dict.fromkeys(v2.LABELS,5)):
                raise ValueError("Physical sitting class counts differ from protocol.")
            if training.metric_set(p[indices],y[indices]) != result["sittings"][str(sitting)]:
                raise ValueError("Sitting metrics do not reproduce.")
        vectors += len(guesses)
        false = [g for g in guesses if g["actual"] in ("NO_COMMAND","STOP") and g["predicted"] in v2.MOVEMENT]
        diagnostics.setdefault(condition,[]).append({"seed":seed,
            "up_prompt_style": [{"repeat":j,"prompted_style":test.STYLES["UP"][j],
                "predicted_counts":dict(Counter(g["predicted"] for i,g in enumerate(guesses)
                     if g["actual"]=="UP" and records[i]["task_index"]//8==j))} for j in range(5)],
            "false_motion_instances":len(false), "false_motion_score_ge_08":sum(max(g["scores"])>=.8 for g in false),
            "false_motion_score_ge_095":sum(max(g["scores"])>=.95 for g in false),
            "maximum_false_motion_score":max((max(g["scores"]) for g in false),default=None)})
    if test.summary(trials) != training.read_json(output/"summary.json"):
        raise ValueError("Primary comparison or aggregation differs.")
    analysis = output/"analysis"
    analysis.mkdir(exist_ok=False)
    write_json(analysis/"verification.json", {"created_utc":v2.utc_now(), "checkpoints":len(expected),
        "saved_score_vectors_reproduced":vectors, "raw_recording_hashes_verified":len(records),
        "original_artifacts_sha256_verified":True, "all_candidate_pooled_and_sitting_metrics_verified":True,
        "scope":"Reproduction of archived results/features, no raw landmark re-access, fitting, tuning or new test pool."})
    write_json(analysis/"exploratory_diagnostics.json", {"scope":"Post-primary descriptive inspection of saved predictions; prompted style is not independently verified performed style. Threshold counts do not measure the live gate.",
        "conditions":diagnostics})
    # Existing plotting library; keep cache within the workspace.
    os.environ["MPLCONFIGDIR"] = str(analysis/"plot_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conditions = training.CONDITIONS
    seeds = sorted({r["seed"] for r in trials})
    fig,axs = plt.subplots(1,2,figsize=(12.5,4.6),gridspec_kw={"width_ratios":[1,1.9]})
    for seed in seeds:
        axs[0].plot(range(3),[by_key[c,seed]["pooled"]["macro_f1"] for c in conditions],
                    marker="o",alpha=.65,lw=1,label=f"Seed {seed}")
    axs[0].plot(range(3),[np.mean([by_key[c,s]["pooled"]["macro_f1"] for s in seeds]) for c in conditions],
                color="black",lw=2.4,marker="D",label="Mean")
    axs[0].set(xticks=range(3),xticklabels=["Reference","Routine","Targeted"],ylabel="Pooled macro F1",
               ylim=(.70,.90),title="All five fixed seeds")
    axs[0].grid(axis="y",alpha=.2);axs[0].legend(fontsize=8,ncol=2,loc="lower right")
    recall = np.array([[100*np.mean([by_key[c,s]["pooled"]["per_class"][label]["recall"] for s in seeds])
                        for label in v2.LABELS] for c in conditions])
    im = axs[1].imshow(recall,cmap="YlGnBu",vmin=0,vmax=100,aspect="auto")
    axs[1].set(xticks=range(8),xticklabels=["FWD","BACK","LEFT","RIGHT","STOP","NONE","UP","DOWN"],
               yticks=range(3),yticklabels=["Reference","Routine","Targeted"],title="Mean class recall (%) · 10 clips per class")
    for j in range(3):
        for i in range(8):
            axs[1].text(i,j,f"{recall[j,i]:.0f}",ha="center",va="center",color="white" if recall[j,i]>=65 else "black",fontsize=10)
    fig.colorbar(im,ax=axs[1],fraction=.032,pad=.025)
    fig.suptitle("Experiment 005 · 80 fresh clips · frozen supervised models",fontsize=14)
    fig.text(.5,.015,"Two sittings of one user; seeds share clips. Raw clip recognition, not robot actions or an unseen-person benchmark.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.94))
    figure = PROJECT/"research/findings/005_boundary_test.png"
    fig.savefig(figure,dpi=180);plt.close(fig)
    write_json(analysis/"derived_artifacts.json",{"report_source_sha256":sha256(Path(__file__)),
        "figure":str(figure),"figure_sha256":sha256(figure),"matplotlib":matplotlib.__version__,
        "verification_sha256":sha256(analysis/"verification.json"),
        "exploratory_diagnostics_sha256":sha256(analysis/"exploratory_diagnostics.json")})
    print(f"Verified {len(expected)} fixed models / {vectors} saved score vectors. Figure: {figure}")
    return analysis


if __name__ == "__main__":
    run()
