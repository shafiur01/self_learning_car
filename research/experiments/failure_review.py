"""Verified development error analysis and an offline landmark review viewer.

Build: .venv/bin/python -m research.experiments.failure_review build
Serve latest: .venv/bin/python -m research.experiments.failure_review serve
Feedback stays separate from recordings, weights and training labels.
"""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import commands_v3_train as training
from . import temporal_comparison as comparison
from .run import PROJECT, sha256, write_json

OUTPUT = PROJECT / "research/experiments/runs/failure_review"
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "review_assets"
SOURCES = (Path(__file__), ASSETS / "viewer.html", ASSETS / "viewer.js",
           HERE / "test_failure_review.py", PROJECT / "research/FAILURE_REVIEW_PROTOCOL.md")


def signals(x, times, visible):
    """Descriptive feature units; not intended-command, metric velocity or depth."""
    return {
        "raw_frames": len(times), "duration_seconds": float(times[-1] - times[0]),
        "tracking_fraction": float(np.mean(visible)), "max_gap_seconds": float(np.diff(times).max()),
        "wrist_endpoint_xy_palm_units": (x[-1, 63:65] - x[0, 63:65]).tolist(),
        "wrist_path_palm_units": float(np.linalg.norm(np.diff(x[:, 63:65], axis=0), axis=1).sum()),
        "pose_variation_rms": float(np.sqrt(np.mean((x[:, :63] - x[:, :63].mean(0)) ** 2))),
        "log_size_endpoint": float(x[-1, 65] - x[0, 65]),
        "log_size_range": float(np.ptp(x[:, 65])),
        "at_feature_clip_limit": int(np.count_nonzero(np.abs(x) >= 10)),
        "motion_or_shape_defined": bool(np.any(visible))}


def nearest_profiles(x, train_x, train_y, labels):
    """Fixed pose-profile RMS rankings; wrist distance reported separately."""
    pose = np.sqrt(np.mean((train_x[:, :, :63].astype(np.float64) - x[None, :, :63]) ** 2, axis=(1, 2)))
    wrist = np.sqrt(np.mean((train_x[:, :, 63:65].astype(np.float64) - x[None, :, 63:65]) ** 2, axis=(1, 2)))
    result = {}
    for index, label in enumerate(labels):
        indices = np.flatnonzero(train_y == index)
        if not len(indices):
            raise ValueError("Missing training class for review comparison.")
        choice = int(indices[np.argmin(pose[indices])])
        result[label] = {"training_row": choice, "pose_profile_rms": float(pose[choice]),
                         "wrist_profile_rms": float(wrist[choice])}
    return result


def load_replay(row, contract, expected_x):
    if row["split"] not in ("train", "validation"):
        raise ValueError("Only development replays allowed; no foreign payload opened.")
    if sha256(row["file"]) != row["sha256"]:
        raise ValueError("Pinned replay changed.")
    xyz, times, visible, meta = training.load_clip(row["file"], contract)
    if any(meta[k] != row[k] for k in ("label", "session", "split")):
        raise ValueError("Replay metadata differs from pinned roles.")
    x = training.v2.features(xyz, times, visible, meta["aspect"])
    np.testing.assert_array_equal(x, expected_x)
    # A centered raw-frame replay uses the same aspect/palm normalization as the
    # feature extractor, but does not invent interpolated frames across gaps.
    centered = xyz.copy()
    centered[:, :, (0, 2)] *= meta["aspect"]
    wrist = centered[:, 0:1].copy()
    scale = np.maximum(np.linalg.norm(centered[:, 9, :2] - wrist[:, 0, :2], axis=1), .02)
    centered = (centered - wrist) / scale[:, None, None]
    if sha256(row["file"]) != row["sha256"]:
        raise ValueError("Replay changed while reading.")
    return {
        "id": Path(row["file"]).stem, "filename": Path(row["file"]).name,
        "file": row["file"], "sha256": row["sha256"], "label": row["label"],
        "session": row["session"], "split": row["split"],
        "hint": meta.get("task_hint", "No recording prompt saved for this training example."),
        "task_index": meta.get("task_index"), "up_shape": meta.get("up_shape"),
        "aspect": meta["aspect"], "times": (times - times[0]).tolist(),
        "visible": visible.tolist(), "xyz": xyz.tolist(), "centered_xyz": centered.tolist(),
        "signals": signals(x, times, visible)}


def assemble(study):
    """Read only the pinned development manifest after weight/prediction verification."""
    study = Path(study)
    verified = comparison.verify(study)
    config = comparison.fitting.read_json(study / "config.json")
    if comparison.fitting.read_json(study / "status.json") != {
            "state": "complete", "completed_trials": len(config["conditions"]) * len(config["training"]["seeds"])}:
        raise ValueError("Require a completed comparison study.")
    manifest = comparison.fitting.read_json(study / "manifest.json")
    if any(r["split"] not in ("train", "validation") for r in manifest["records"]):
        raise ValueError("Review cohort must contain only development inputs.")
    by_file = {r["file"]: r for r in manifest["records"]}
    with np.load(study / "development_features.npz", allow_pickle=False) as a:
        tx, ty, vx, vy = (a[k].copy() for k in ("train_x", "train_y", "validation_x", "validation_y"))
    predictions = {}
    for family in config["conditions"]:
        for seed in config["training"]["seeds"]:
            for row in comparison.fitting.read_json(study / f"{family}_seed_{seed}/validation_predictions.json"):
                predictions.setdefault(row["file"], []).append({
                    "family": family, "seed": seed, "predicted": row["predicted"],
                    "score": max(row["scores"]), "scores": row["scores"],
                    "wrong": row["predicted"] != row["actual"]})
    clips, references, summaries = [], {}, {}
    labels = list(training.v2.LABELS)
    for i, file in enumerate(manifest["rows"]["validation"]):
        row = by_file[file]
        if row["split"] != "validation" or row["label"] != labels[int(vy[i])]:
            raise ValueError("Validation row/label changed.")
        clip = load_replay(row, manifest["contract"], vx[i])
        clip["predictions"] = predictions[file]
        clip["wrong_count"] = sum(p["wrong"] for p in predictions[file])
        clip["nearest_by_label"] = nearest_profiles(vx[i], tx, ty, labels) if clip["signals"]["motion_or_shape_defined"] else {}
        wrong = Counter(p["predicted"] for p in predictions[file] if p["wrong"])
        common_wrong = min(wrong, key=lambda label: (-wrong[label], labels.index(label))) if wrong else None
        clip["neighbours"] = {}
        for kind, label in (("same_label", clip["label"]), ("confused_label", common_wrong)):
            if label is None or label not in clip["nearest_by_label"]:
                continue
            nearest = clip["nearest_by_label"][label]
            index = nearest["training_row"]
            training_row = by_file[manifest["rows"]["train"][index]]
            if training_row["split"] != "train":
                raise ValueError("Nearest reference must remain training data.")
            ref_id = Path(training_row["file"]).stem
            if ref_id not in references:
                references[ref_id] = load_replay(training_row, manifest["contract"], tx[index])
            clip["neighbours"][kind] = {"id": ref_id, "label": label,
                "pose_profile_rms": nearest["pose_profile_rms"], "wrist_profile_rms": nearest["wrist_profile_rms"]}
        clips.append(clip)
    for label in labels:
        group = [c for c in clips if c["label"] == label]
        summaries[label] = {"validation_clips": len(group), "unique_any_model_failures": sum(c["wrong_count"] > 0 for c in group),
            "repeated_model_errors": sum(c["wrong_count"] for c in group),
            "unique_failures_by_family": {f: sum(any(p["wrong"] for p in c["predictions"] if p["family"] == f)
                                             for c in group) for f in config["conditions"]}}
    payload = {"version": 1, "study": str(study.resolve()), "study_manifest_sha256": sha256(study / "manifest.json"),
        "families": config["conditions"], "seeds": config["training"]["seeds"], "labels": labels,
        "clips": clips, "references": references, "summary": summaries,
        "scope": "Inspected development landmark review; no RGB, label changes, retraining or robot evaluation."}
    identity = json.dumps({"study_manifest": payload["study_manifest_sha256"],
        "models": verified["checkpoints_sha256"], "recordings": {r["file"]: r["sha256"] for r in manifest["records"]}}, sort_keys=True)
    payload["bundle_id"] = hashlib.sha256(identity.encode()).hexdigest()
    return payload, verified


def render_html(payload):
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = (ASSETS / "viewer.html").read_text()
    return template.replace("__REVIEW_DATA__", encoded).replace("__REVIEW_JS__", (ASSETS / "viewer.js").read_text())


def build(study=comparison.OUTPUT / "20261014T011524632211Z", output_root=OUTPUT):
    payload, verified = assemble(study)
    output = Path(output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "review_data.json", payload)
    write_json(output / "verification.json", verified)
    (output / "review.html").write_text(render_html(payload))
    with (output / "failures.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "original_label", "family", "seed", "predicted", "score", "session"])
        writer.writeheader()
        for clip in payload["clips"]:
            for prediction in clip["predictions"]:
                if prediction["wrong"]:
                    writer.writerow({k: {**clip, **prediction, "original_label": clip["label"]}[k] for k in writer.fieldnames})
    analysis = {"summary": payload["summary"], "clips": [{k: c[k] for k in
        ("id", "filename", "label", "session", "hint", "wrong_count", "signals", "nearest_by_label", "neighbours", "predictions")} for c in payload["clips"]],
        "scope": "Fixed descriptive features/neighbours; not causal explanations or automatic relabeling."}
    write_json(output / "analysis.json", analysis)
    for path in SOURCES:
        destination = output / "source" / path.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    hashes = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}
    write_json(output / "artifact_sha256.json", hashes)
    write_json(output / "status.json", {"state": "complete", "validation_clips": len(payload["clips"]),
        "unique_failures": sum(c["wrong_count"] > 0 for c in payload["clips"]), "human_review": "pending"})
    Path(output_root).mkdir(parents=True, exist_ok=True)
    write_json(Path(output_root) / "latest.json", {"folder": str(output.resolve()), "bundle_id": payload["bundle_id"]})
    print(f"Review ready: {output / 'review.html'}")
    print("Original categories: unique failing clips / validation clips")
    for label, row in payload["summary"].items():
        print(f"{label}: {row['unique_any_model_failures']}/{row['validation_clips']}")
    return output


def import_feedback(path, review):
    """Versioned observations only. No original label or recording is modified."""
    review = Path(review)
    payload = comparison.fitting.read_json(review / "review_data.json")
    incoming = comparison.fitting.read_json(path)
    if incoming.get("version") != 1 or incoming.get("bundle_id") != payload["bundle_id"]:
        raise ValueError("Feedback belongs to a different review bundle.")
    by_id = {c["id"]: c for c in payload["clips"]}
    allowed = {"matches", "ambiguous", "tracking", "possible_label_error"}
    seen = set()
    for note in incoming.get("reviews", []):
        clip = by_id.get(note.get("clip_id"))
        if (clip is None or clip["id"] in seen or note.get("clip_sha256") != clip["sha256"]
            or note.get("original_label") != clip["label"] or note.get("assessment") not in allowed
            or note.get("suggested_label") not in (None, "", *payload["labels"])
            or not isinstance(note.get("notes"), str)):
            raise ValueError("Invalid, duplicate or unpinned human review.")
        if sha256(clip["file"]) != clip["sha256"]:
            raise ValueError("Original recording changed before review import.")
        seen.add(clip["id"])
    if not seen:
        raise ValueError("No completed human observations to import.")
    destination = review / "human_reviews" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, {"imported_utc": datetime.now(timezone.utc).isoformat(),
        "export_file_sha256": sha256(path), "feedback": incoming,
        "scope": "Human observations/suggested labels, unapplied; review effort counts in future budgets."})
    print(f"Saved {len(seen)} observations separately: {destination}")
    return destination


def latest():
    return Path(comparison.fitting.read_json(OUTPUT / "latest.json")["folder"])


def serve(review, port=8765):
    handler = partial(SimpleHTTPRequestHandler, directory=str(Path(review).resolve()))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"Open http://127.0.0.1:{port}/review.html — Ctrl+C stops the viewer server.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--study", type=Path, default=comparison.OUTPUT / "20261014T011524632211Z")
    server_parser = sub.add_parser("serve")
    server_parser.add_argument("--review", type=Path)
    server_parser.add_argument("--port", type=int, default=8765)
    feedback_parser = sub.add_parser("import-notes")
    feedback_parser.add_argument("file", type=Path)
    feedback_parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if args.action == "build":
        build(args.study)
    elif args.action == "serve":
        serve(args.review or latest(), args.port)
    else:
        import_feedback(args.file, args.review or latest())


if __name__ == "__main__":
    main()
