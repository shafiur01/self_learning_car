"""Describe already completed experiment012 scores; never fit or re-evaluate a pool.

Run once: .venv/bin/python -m research.experiments.prospective_motion_analysis
Adds derived summaries, tables, plots and landmark review; frozen sources stay intact.
"""

import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import failure_review as review
from . import prospective_motion as prospective
from .run import PROJECT, sha256, write_json

FAMILIES = ("cnn", "lstm", "bilstm", "local_set", "feature_set")
MODES = ("camera_pose", "camera_motion", "hand_pose", "hand_motion")
NAMES = {"cnn": "CNN", "lstm": "LSTM", "bilstm": "BiLSTM", "local_set": "Local set", "feature_set": "Feature set", "causal_cnn": "Causal CNN"}
MODE_NAMES = {"camera_pose": "Camera position", "camera_motion": "Camera + motion", "hand_pose": "Hand position", "hand_motion": "Hand + motion"}


def read(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_summary(rows, scope):
    metrics = [r["pooled"] if scope == "pooled" else r["sittings"][scope] for r in rows]
    keys = ("accuracy", "macro_f1", "nll", "stop_recall", "no_command_to_motion_rate", "no_command_to_down_rate", "stop_to_motion_count")
    return {"metrics": {k: {"mean": float(np.mean([r[k] for r in metrics])),
                           "seed_sd": float(np.std([r[k] for r in metrics], ddof=1))} for k in keys},
            "class_recall": {label: float(np.mean([r["per_class"][label]["recall"] for r in metrics])) for label in metrics[0]["labels"]}}


def up_style_report(output, payload):
    """Post-hoc description only; four distinct recordings per style, no selection."""
    rows = []
    for family in FAMILIES:
        for style in ("index_up", "two_fingers_up", "thumbs_up"):
            clips = [c for c in payload["clips"] if c["label"] == "UP" and c["up_shape"] == style]
            predictions = [r for c in clips for r in c["predictions"] if r["family"] == family + "/camera_motion/standard"]
            correct = sum(not r["wrong"] for r in predictions)
            rows.append({"family": family, "style": style, "unique_clips": len(clips),
                         "correct_of_20_repeated_predictions": correct, "repeated_predictions": len(predictions),
                         "recall": correct / len(predictions),
                         "scope": "Post-hoc descriptive; four unique clips/style from one user; not an independent primary contrast."})
    csv_write(Path(output) / "up_style_posthoc.csv", rows)


def replay(row, expected, contract):
    """Post-access replay; verify raw and version2 features before displaying points."""
    if sha256(row["file"]) != row["sha256"]:
        raise ValueError("Pinned replay changed.")
    xyz, times, visible, meta = prospective.experiment.motion.previous.training.load_clip(row["file"], contract)
    x, audits = prospective.experiment.motion.rep.features(xyz, times, visible, meta["aspect"], "camera_motion")
    np.testing.assert_array_equal(x, expected)
    points = xyz.astype(np.float64).copy()
    points[:, :, (0, 2)] *= meta["aspect"]
    scale = np.maximum(np.linalg.norm(points[:, 9] - points[:, 0], axis=1), .02)
    centered = (points - points[:, 0:1]) / scale[:, None, None]
    result = {"id": Path(row["file"]).stem, "filename": Path(row["file"]).name,
              "file": row["file"], "sha256": row["sha256"], "label": row["label"],
              "session": row["session"], "split": meta["split"], "sitting": row.get("sitting"),
              "hint": meta.get("task_hint", "Original training recording; no prompt saved."),
              "task_index": meta.get("task_index"), "up_shape": meta.get("up_shape"),
              "aspect": meta["aspect"], "times": (times - times[0]).tolist(), "visible": visible.tolist(),
              "xyz": xyz.tolist(), "centered_xyz": centered.tolist(),
              "signals": {"raw_frames": len(times), "duration_seconds": float(times[-1] - times[0]),
                  "tracking_fraction": float(visible.mean()), "max_gap_seconds": float(np.diff(times).max()),
                  "wrist_path_palm_units": float(np.linalg.norm(np.diff(x[:, 63:65], axis=0), axis=1).sum()),
                  "pose_variation_rms": float(np.sqrt(np.mean((x[:, :63] - x[:, :63].mean(0)) ** 2))),
                  "motion_or_shape_defined": bool(visible.any()), "feature_audit": audits}}
    if sha256(row["file"]) != row["sha256"]:
        raise ValueError("Replay changed during loading.")
    return result


def build(root=prospective.ROOT):
    root = Path(root)
    access = read(root / "evaluation_access.json")
    if access["state"] != "complete":
        raise ValueError("Only analyze the completed joint evaluation; never restart inference.")
    plan = prospective.read_plan(root)
    output = root / "analysis"
    output.mkdir(exist_ok=False)
    evaluation = root / "evaluation"
    manifest, trials = read(evaluation / "manifest.json"), read(evaluation / "trials.json")
    labels = list(prospective.validation.v2.LABELS)
    groups, scores, model_rows, class_rows, error_rows = defaultdict(list), defaultdict(list), [], [], []
    condition_names = {}
    for trial in trials:
        key = "/".join(trial[k] for k in ("family", "mode", "variant"))
        groups[key].append(trial)
        condition_names[key] = f"{NAMES[trial['family']]} · {MODE_NAMES[trial['mode']]}" + (" · no elapsed" if trial["variant"] == "no_elapsed" else "")
        identity = {k: trial[k] for k in ("family", "mode", "variant", "seed")}
        for scope in ("pooled", "1", "2"):
            metrics = trial["pooled"] if scope == "pooled" else trial["sittings"][scope]
            model_rows.append({**identity, "scope": scope, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                              "stop_recall": metrics["stop_recall"], "no_command_to_motion_rate": metrics["no_command_to_motion_rate"], "nll": metrics["nll"]})
            for label, values in metrics["per_class"].items():
                class_rows.append({**identity, "scope": scope, "label": label, **values})
        for row in read(evaluation / trial["folder"] / "predictions.json"):
            prediction = {"family": key, "seed": trial["seed"], "predicted": row["predicted"], "score": max(row["scores"]),
                          "scores": row["scores"], "wrong": row["predicted"] != row["actual"]}
            scores[row["file"]].append(prediction)
            if prediction["wrong"]:
                error_rows.append({**identity, "file": row["file"], "actual": row["actual"], "predicted": row["predicted"], "raw_top_score": prediction["score"]})
    expected_seeds = {11, 22, 33, 44, 55}
    assert all(len(rows) == 5 and {r['seed'] for r in rows} == expected_seeds for rows in groups.values())
    summary = {"groups": {key: {scope: group_summary(rows, scope) for scope in ("pooled", "1", "2")} for key, rows in groups.items()}, "paired_camera_motion_minus_position": {}}
    for family in FAMILIES:
        pose = {r["seed"]: r for r in groups[f"{family}/camera_pose/standard"]}
        motion = {r["seed"]: r for r in groups[f"{family}/camera_motion/standard"]}
        contrast = {}
        for scope in ("pooled", "1", "2"):
            get = lambda r: r["pooled"] if scope == "pooled" else r["sittings"][scope]
            differences = [get(motion[s])["macro_f1"] - get(pose[s])["macro_f1"] for s in sorted(pose)]
            contrast[scope] = {"differences_in_seed_order": differences, "mean": float(np.mean(differences)), "positive_pairs": sum(d > 0 for d in differences)}
        summary["paired_camera_motion_minus_position"][family] = contrast
    csv_write(output / "model_results.csv", model_rows)
    csv_write(output / "per_class_results.csv", class_rows)
    csv_write(output / "full_clip_errors.csv", error_rows)

    # Re-extract all 384 feature arrays after the one-time evaluation, no refitting.
    feature_checks, raw_stats, clips = 0, [], []
    with np.load(evaluation / "features.npz", allow_pickle=False) as arrays:
        assert len(manifest["records"]) == len(arrays["y"]) == 96
        for index, row in enumerate(manifest["records"]):
            xyz, times, visible, meta = prospective.experiment.motion.previous.training.load_clip(row["file"], plan["contract"])
            assert prospective.validation.v2.content_sha(xyz, times, visible, meta["aspect"]) == row["raw_content_sha256"]
            assert row["raw_content_sha256"] not in plan["banned_raw_contents"] and row["session"] not in plan["banned_sessions"]
            assert bool(prospective.validation.v2.quality(times, visible)) == row["passes_tracking_quality"]
            assert labels[int(arrays["y"][index])] == row["label"]
            for mode in MODES:
                actual, audit = prospective.experiment.motion.rep.features(xyz, times, visible, meta["aspect"], mode)
                np.testing.assert_array_equal(actual, arrays[mode][index])
                feature_checks += 1
                if mode == "camera_motion":
                    raw_stats.append({**row, "tracking_fraction": float(visible.mean()), "up_shape": meta.get("up_shape"), **audit})
            clip = replay(row, arrays["camera_motion"][index], plan["contract"])
            clip["hint"] = f"Sitting {row['sitting']} · " + clip["hint"]
            clip["predictions"] = scores[row["file"]]
            assert len(clip["predictions"]) == 115
            clip["wrong_count"] = sum(p["wrong"] for p in clip["predictions"])
            clips.append(clip)

    # Descriptive shape-neighbor references use the same new camera normalization.
    study = Path(plan["study"])
    training_manifest = read(study / "manifest.json")
    by_file = {r["file"]: r for r in training_manifest["records"]}
    references = {}
    with np.load(study / "features.npz", allow_pickle=False) as arrays:
        train_x, train_y = arrays["camera_motion_train_x"], arrays["train_y"]
        with np.load(evaluation / "features.npz", allow_pickle=False) as query:
            for index, clip in enumerate(clips):
                nearest = review.nearest_profiles(query["camera_motion"][index], train_x, train_y, labels) if clip["signals"]["motion_or_shape_defined"] else {}
                clip["nearest_by_label"], clip["neighbours"] = nearest, {}
                wrong = Counter(p["predicted"] for p in clip["predictions"] if p["wrong"])
                confused = min(wrong, key=lambda label: (-wrong[label], labels.index(label))) if wrong else None
                for kind, label in (("same_label", clip["label"]), ("confused_label", confused)):
                    if label not in nearest:
                        continue
                    info = nearest[label]
                    row = by_file[training_manifest["rows"]["train"][info["training_row"]]]
                    assert row["split"] == "train"
                    identity = Path(row["file"]).stem
                    if identity not in references:
                        references[identity] = replay(row, train_x[info["training_row"]], plan["contract"])
                    clip["neighbours"][kind] = {"id": identity, "label": label, "pose_profile_rms": info["pose_profile_rms"], "wrist_profile_rms": info["wrist_profile_rms"]}

    summary["failure_categories"] = {label: {"unique_clips": 12, "unique_any_model_failures": sum(c["wrong_count"] > 0 for c in clips if c["label"] == label),
        "repeated_model_errors": sum(c["wrong_count"] for c in clips if c["label"] == label)} for label in labels}
    summary["unique_failing_clips_any_of_115"] = sum(c["wrong_count"] > 0 for c in clips)
    summary["repeated_errors_all_115"] = len(error_rows)
    summary["scope"] = "All predeclared frozen candidates, full clips, same user/two sittings one evening. Now inspected development. Seed dispersion is not population uncertainty. No new training, gates, robot actions, early-onset or label-saving result."
    write_json(output / "summary.json", summary)
    write_json(output / "raw_feature_audit.json", {"raw_clips": 96, "reextracted_feature_arrays": feature_checks, "raw_stats": raw_stats,
        "training_overlap": 0, "no_hand_clips": sum(r["no_hand"] for r in raw_stats), "retained_quality_flag_failures": sum(not r["passes_tracking_quality"] for r in raw_stats)})
    csv_write(output / "clip_review_index.csv", [{"file": c["file"], "sitting": c["sitting"], "label": c["label"], "up_shape": c["up_shape"],
        "wrong_of_115": c["wrong_count"], "tracking_fraction": c["signals"]["tracking_fraction"], "hint": c["hint"]} for c in clips])
    write_json(output / "costs.json", {"human_labels_added": 96, "new_fits": 0,
        "attempts": [read(p) for p in sorted((root / "attempts").glob("*.json"))],
        "camera_interval_seconds": sum(read(p)["wall_seconds"] for p in (root / "attempts").glob("*.json")),
        "scope": "Capture/replay/preparation/pause interval, not pure annotation. Software work, earlier fits/labels/review/pretraining are additional costs. No label savings demonstrated."})

    payload = {"version": 1, "study": str(evaluation.resolve()), "study_manifest_sha256": sha256(evaluation / "manifest.json"),
        "families": list(groups), "seeds": sorted(expected_seeds), "labels": labels, "clips": clips, "references": references,
        "summary": {label: {"validation_clips": 12, **values} for label, values in summary["failure_categories"].items()},
        "scope": summary["scope"]}
    payload["bundle_id"] = hashlib.sha256(json.dumps({"freeze": sha256(root / "freeze.json"), "manifest": payload["study_manifest_sha256"],
        "prediction_archives": {t["folder"]: sha256(evaluation / t["folder"] / "predictions.json") for t in trials}}, sort_keys=True).encode()).hexdigest()
    write_json(output / "review_data.json", payload)
    up_style_report(output, payload)
    html, js = (review.ASSETS / name for name in ("viewer.html", "viewer.js"))
    template, script = html.read_text(), js.read_text()
    script, count = re.subn(r"const FAMILY_NAMES = .*?;", lambda _: "const FAMILY_NAMES = " + json.dumps(condition_names) + ";", script, count=1)
    assert count == 1
    script = script.replace('el("category").value="STOP"', 'el("category").value="UP"').replace('"All five families"', '"All 23 model conditions"')
    script = script.replace('DATA.families.forEach(family=>', 'DATA.families.filter(f=>el("family").value==="all"||f===el("family").value).forEach(family=>')
    script = script.replace('const values=current.predictions.filter(p=>p.family===family)', 'const values=selectedPredictions(current).filter(p=>p.family===family)')
    script = script.replace('Validation session: ', 'Recording session: ')
    template = template.replace('Model family<select', 'Model condition<select').replace('Each family has five separately trained models.', 'Each condition has five frozen training seeds. All 115 models were jointly evaluated before this review.')
    template = template.replace('Existing development validation · original labels', 'Experiment012: new sessions now inspected · original labels')
    template = template.replace('Watch the recorded hand points. Compare examples. Add your observations.', 'Two new sittings · all 115 frozen models · inspect UP and NO_COMMAND errors first.')
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    (output / "review.html").write_text(template.replace("__REVIEW_DATA__", encoded).replace("__REVIEW_JS__", script))
    (output / "viewer.js").write_text(script)
    source = output / "source"
    source.mkdir()
    for path in (Path(__file__), Path(review.__file__), html, js):
        target = source / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (PROJECT / "research/experiments/prospective_motion_review.html").write_text('<!doctype html><meta charset="utf-8"><title>New-session gesture review</title><p><a href="data/prospective_motion_011/analysis/review.html">Open experiment012 gesture replays</a></p>')
    plot(output, summary, read(study / "summary.json"))
    write_json(output / "artifact_sha256.json", {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file() and "plot_cache" not in p.parts})
    print(json.dumps({"output": str(output), "models": len(trials), "raw_feature_arrays_verified": feature_checks,
        "unique_failed_clips": summary["unique_failing_clips_any_of_115"], "failure_categories": summary["failure_categories"], "training_references": len(references)}, indent=2))
    return output


def plot(output, summary, old):
    os.environ.setdefault("MPLCONFIGDIR", str(output / "plot_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    matrix = np.array([[summary["groups"][f"{f}/{m}/standard"]["pooled"]["metrics"]["accuracy"]["mean"] for m in MODES] for f in FAMILIES])
    ax = axes[0, 0]
    image = ax.imshow(matrix, vmin=.8, vmax=1, cmap="Blues", aspect="auto")
    ax.set(xticks=range(4), xticklabels=["Camera\nposition", "Camera\nmotion", "Hand\nposition", "Hand\nmotion"], yticks=range(5), yticklabels=[NAMES[f] for f in FAMILIES], title="A  New-session mean accuracy\nFive frozen seeds per cell")
    for i in range(5):
        for j in range(4):
            ax.text(j, i, f"{matrix[i,j]*100:.2f}%", ha="center", va="center", color="white" if matrix[i,j] > .94 else "#173c4c")
    figure.colorbar(image, ax=ax, shrink=.75, label="Accuracy")
    ax = axes[0, 1]
    for i, f in enumerate(FAMILIES):
        d = np.array(summary["paired_camera_motion_minus_position"][f]["pooled"]["differences_in_seed_order"]) * 100
        ax.bar(i, d.mean(), color="#5d8da6" if d.mean() > 0 else "#bc7c86")
        ax.scatter(i + np.linspace(-.12, .12, 5), d, color="#203a47", s=25, zorder=3)
    ax.axhline(0, color="#60717b", lw=1)
    ax.set(xticks=range(5), xticklabels=[NAMES[f] for f in FAMILIES], ylabel="Paired macro F1 difference × 100", title="B  Camera motion minus camera position\nDots show paired seeds; negatives retained")
    ax = axes[1, 0]
    x = np.arange(5)
    for offset, scope, name, color in ((-.14, "1", "Sitting 1", "#6b94ab"), (.14, "2", "Sitting 2", "#bd8390")):
        ax.bar(x + offset, [summary["groups"][f"{f}/camera_motion/standard"][scope]["metrics"]["accuracy"]["mean"] * 100 for f in FAMILIES], .28, label=name, color=color)
    ax.set(xticks=x, xticklabels=[NAMES[f] for f in FAMILIES], ylim=(0, 105), ylabel="Accuracy (%)", title="C  Camera-motion results by sitting\n48 unique clips per sitting")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax = axes[1, 1]
    for f in FAMILIES:
        earlier = old["groups"][f"{f}/camera_motion"]["metrics"]["accuracy"]["mean"] * 100
        fresh = summary["groups"][f"{f}/camera_motion/standard"]["pooled"]["metrics"]["accuracy"]["mean"] * 100
        ax.plot([0, 1], [earlier, fresh], marker="o", label=NAMES[f])
    ax.set(xticks=[0, 1], xticklabels=["Inspected development48", "New frozen evaluation96"], ylim=(80, 101), ylabel="Accuracy (%)", title="D  Earlier and new cohorts\nDifferent clips: descriptive generalization gap")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    figure.suptitle("Experiment 012: fresh-session evidence is mixed", fontsize=17)
    figure.supxlabel("One user · two sittings on one evening · no retraining · no population, label-saving or robot-control claim", fontsize=10)
    figure.savefig(output / "figure.png", dpi=170)
    figure.savefig(output / "figure.pdf")
    plt.close(figure)


if __name__ == "__main__":
    build()
