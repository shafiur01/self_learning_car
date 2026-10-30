"""Prepare/freeze/fit/retrospectively evaluate pilot014; never activate a model.

Each action is explicit. Learning bundles contain no hidden training labels.
Evaluation and oracle diagnostics are unavailable until all candidates finish.
"""

import argparse
import json
import platform
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import cross_model_motion as old
from . import evidence_profile as ep
from .run import PROJECT, sha256, write_json

ID = "014_evidence_profile_budget160"
SOURCE = PROJECT / "research/experiments/runs/cross_model_motion/20261015T164901242592Z"
ENDPOINT = PROJECT / "research/experiments/data/prospective_motion_011"
OUTPUT = PROJECT / "research/experiments/runs/evidence_profile_budget160"
LABELS = old.motion.previous.v2.LABELS
GROUP_A = {"20260930T062558-33e35afb", "20261008T021615-a0b59124", "20261012T174044-0244fa51"}
METHODS = ("supervised160", "supervised160_long", "selftrain_full_conf", "selftrain_subset_conf",
           "profile", "embedding", "matched_conf", "matched_random")
SOURCES = tuple(dict.fromkeys((*old.SOURCES, "research/experiments/evidence_profile_demo.py",
    "research/experiments/evidence_profile.py", "research/experiments/evidence_profile_run.py",
    "research/experiments/evidence_profile_audit.py",
    "research/experiments/test_evidence_profile.py", "research/EVIDENCE_PROFILE_PROTOCOL.md",
    "research/experiments/evidence_profile_budget_run.py",
    "research/experiments/evidence_profile_budget_audit.py",
    "research/experiments/test_evidence_profile_budget.py", "research/EVIDENCE_PROFILE_BUDGET_PROTOCOL.md")))

PRIOR = PROJECT / "research/experiments/runs/evidence_profile/20261016T030844123147Z"


def read(path):
    return json.loads(Path(path).read_text())


def sample_rows(y, groups, seed, per_class_group):
    """Same fixed stratified permutations at either budget; never style-resample."""
    rng = np.random.default_rng(seed + 4_000_000)
    sides = []
    for side in (0, 1):
        selected = []
        for c in range(8):
            candidates = np.flatnonzero((groups == side) & (y == c))
            if len(candidates) < per_class_group:
                raise ValueError("Insufficient command support in a whole-session group.")
            selected.extend(rng.permutation(candidates)[:per_class_group].tolist())
        sides.append(np.array(selected, dtype=np.int64))
    labeled = np.r_[sides[0], sides[1]]
    unlabeled = np.setdiff1d(np.arange(len(y)), labeled)
    if not np.array_equal(np.bincount(y[labeled], minlength=8), np.full(8, 2 * per_class_group)):
        raise ValueError("All eight commands must have equal revealed-label counts.")
    return sides, labeled, unlabeled


def coverage_report(rows, samples, allowed):
    report = []
    for seed, sample in samples.items():
        styles = []
        for name in ('teacher_a', 'teacher_b'):
            selected = [rows[i] for i in sample[name] if rows[i]['label'] == 'UP']
            values = sorted({r.get('up_shape') for r in selected if r.get('up_shape')})
            styles.append(values)
        report.append({'seed': int(seed), 'teacher_up_shapes': styles,
                       'both_teachers_all_up_shapes': all(set(v) == set(allowed) for v in styles)})
    return {'role': 'Metadata-only coverage audit; no style-based reselection.', 'rows': report}


def prepare():
    old.verify(SOURCE)
    from . import evidence_profile_run as parent
    parent_config = parent.frozen(PRIOR, endpoint=True)
    if read(PRIOR / 'status.json')['phase'] != 'complete':
        raise ValueError("Require the completed80-label pilot as the parent.")
    manifest = read(SOURCE / "manifest.json")
    rows = [r for r in manifest["records"] if r["split"] == "train" and r["source"] == "recording"]
    if len(rows) != 405 or len({r['raw_content_sha256'] for r in rows}) != 405:
        raise ValueError("Require405 unique original training recordings.")
    endpoint = read(ENDPOINT / "evaluation/manifest.json")
    if len(endpoint["records"]) != 96:
        raise ValueError("Require existing96 retrospective endpoint records.")
    ts, ids = {r['session'] for r in rows}, {r['raw_content_sha256'] for r in rows}
    if any(r['session'] in ts or r['raw_content_sha256'] in ids for r in endpoint['records']):
        raise ValueError("Endpoint overlaps training.")
    output = OUTPUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (output / "learner").mkdir(parents=True, exist_ok=False)
    config = {"experiment_id": ID, "seeds": [11, 22, 33, 44, 55], "methods": list(METHODS),
        "training": {"steps": 560, "validation_checks": 28, "batch_size": 32, "learning_rate": .001, "ramp_steps": 140},
        "selector": {"confidence": .95, "anchor_confidence": .5, "minimum_anchors": 3,
                     "scale_floor": .05, "radius_floor_squared": .05**2},
        "source_study": str(SOURCE), "endpoint": str(ENDPOINT), "parameters": ep.PARAMETERS,
        "features": 149, "feature_schema": "motion_v2_availability_v1", "threads": 2,
        "labeled_budget": 160, "hidden_budget": 245, "teacher_labels_per_class": 10,
        "parent_study": str(PRIOR), "intervention": "Nested labeled budget80 to160; all learning and selector rules unchanged.",
        "planned_fits": 65, "contract": manifest["contract"],
        "scope": "Fixed-data retrospective single-user mechanism pilot; not untouched testing or label/compute saving evidence."}
    for key in ('seeds', 'training', 'selector', 'source_study', 'endpoint', 'parameters',
                'features', 'feature_schema', 'threads', 'planned_fits', 'contract'):
        if config[key] != parent_config[key]:
            raise ValueError(f"Budget-only intervention changed {key}.")
    if read(PRIOR / 'registry.json')['records'] != rows:
        raise ValueError("Original training rows changed from the80-label run.")
    write_json(output / "config.json", config)
    position = {file: i for i, file in enumerate(manifest['rows']['train'])}
    with np.load(SOURCE / "features.npz", allow_pickle=False) as a:
        x = a['camera_motion_train_x'][[position[r['file']] for r in rows]].copy()
        y = np.array([LABELS.index(r['label']) for r in rows], dtype=np.int64)
        np.savez_compressed(output / "selection.npz", x=a['camera_motion_validation_x'], y=a['validation_y'])
    np.savez_compressed(output / "oracle.npz", y=y)
    np.savez_compressed(output / "reference405.npz", x=x, y=y)
    write_json(output / "registry.json", {"records": rows, "selection_files": manifest['rows']['validation'],
        "source_manifest_sha256": sha256(SOURCE / "manifest.json"), "scope": "Sampler/oracle provenance, not learner input."})
    groups = np.array([0 if r['session'] in GROUP_A else 1 for r in rows])
    samples = {}
    for seed in config['seeds']:
        sides, labeled, unlabeled = sample_rows(y, groups, seed, 10)
        previous = read(PRIOR / "sampling.json")[str(seed)]
        if not set(previous['revealed_rows']).issubset(labeled):
            raise ValueError("Original80 labels must be nested in the160-label selection.")
        file = output / "learner" / f"seed_{seed}.npz"
        np.savez_compressed(file, x=x, labeled=labeled, y=y[labeled], teacher_a=sides[0], teacher_b=sides[1], unlabeled=unlabeled)
        ep.LearnerPool.load(file)
        samples[str(seed)] = {"revealed_rows": labeled.tolist(), "teacher_a": sides[0].tolist(), "teacher_b": sides[1].tolist(),
            "hidden_rows": unlabeled.tolist(), "revealed_class_counts": np.bincount(y[labeled], minlength=8).tolist()}
    write_json(output / "sampling.json", samples)
    write_json(output / "coverage.json", coverage_report(rows, samples, config['contract']['allowed_up_shapes']))
    write_json(output / "dataset_summary.json", {"original_training": len(rows), "class_counts": dict(Counter(r['label'] for r in rows)),
        "labeled_per_seed": 160, "hidden_per_seed": 245, "selection": 48, "retrospective_endpoint": 96,
        "group_sizes": np.bincount(groups).tolist(), "excluded_model_selected_corrections": 133,
        "new_recordings": 0, "training_plus_selection_labels": 208})
    inputs = [p for p in output.rglob('*') if p.is_file()]
    freeze = {"experiment_id": ID, "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {str(p.relative_to(output)): sha256(p) for p in inputs},
        "external": {str(SOURCE / name): sha256(SOURCE / name) for name in ('features.npz', 'manifest.json')},
        "endpoint_inputs": {str(ENDPOINT / 'evaluation' / name): sha256(ENDPOINT / 'evaluation' / name) for name in ('features.npz', 'manifest.json')},
        "raw_inputs": {r['file']: r['sha256'] for r in manifest['records']},
        "endpoint_raw": {r['file']: r['sha256'] for r in endpoint['records']},
        "parent_artifacts": {str(p): sha256(p) for p in PRIOR.rglob('*')
                             if p.is_file() and 'plot_cache' not in p.parts},
        "source_sha256": {name: sha256(PROJECT / name) for name in SOURCES},
        "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform()}}
    for name in SOURCES:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(PROJECT / name, target)
    write_json(output / 'freeze.json', freeze)
    (output / 'freeze.sha256').write_text(sha256(output / 'freeze.json') + '\n')
    write_json(output / 'status.json', {"phase": "prepared", "completed_fits": 0})
    write_json(output / 'trials.json', [])
    write_json(OUTPUT / 'latest.json', {"study": str(output)})
    print(json.dumps({"study": str(output), "original_training": 405, "labeled": 160, "unlabeled": 245, "commands": list(LABELS), "planned_fits": 65}, indent=2))
    return output


def frozen(study, endpoint=False):
    study = Path(study)
    if sha256(study / 'freeze.json') != (study / 'freeze.sha256').read_text().strip():
        raise ValueError("Frozen experiment policy changed.")
    f = read(study / 'freeze.json')
    for name, digest in f['inputs'].items():
        if sha256(study / name) != digest:
            raise ValueError("Prepared input changed.")
    for name, digest in f['source_sha256'].items():
        if sha256(study / 'source' / name) != digest or (name.endswith('.py') and sha256(PROJECT / name) != digest):
            raise ValueError("Frozen executable source changed.")
    for group in ('external', 'raw_inputs', 'parent_artifacts', *(('endpoint_inputs', 'endpoint_raw') if endpoint else ())):
        if any(sha256(Path(p)) != digest for p, digest in f[group].items()):
            raise ValueError("Pinned source/recording changed.")
    return read(study / 'config.json')


def load_model(path):
    c = torch.load(path, map_location='cpu', weights_only=True)
    expected = {"experiment_id": ID, "features": 149, "frames": 32, "labels": list(LABELS), "feature_schema": "motion_v2_availability_v1", "parameters": ep.PARAMETERS}
    if any(c.get(k) != v for k, v in expected.items()):
        raise ValueError("Not a compatible research-only mask-aware checkpoint.")
    model = ep.Model(); model.load_state_dict(c['state_dict']); model.eval()
    return model, c


def save_fit(study, name, seed, model, history, stats, vx, vy):
    folder = Path(study) / name
    folder.mkdir(exist_ok=False)
    logits, _ = ep.infer(model, vx)
    p = ep.probabilities(logits[0])
    row = {"folder": name, "seed": seed, **stats, "validation": old.motion.previous.fitting.metric_set(p, vy), "state_sha256": ep.state_hash(model)}
    torch.save({"experiment_id": ID, "features": 149, "frames": 32, "labels": list(LABELS),
        "feature_schema": "motion_v2_availability_v1", "parameters": ep.PARAMETERS, "seed": seed, "folder": name,
        "config_sha256": sha256(Path(study) / 'config.json'), "freeze_sha256": sha256(Path(study) / 'freeze.json'),
        "best_step": stats['best_step'], "state_dict": model.state_dict()}, folder / 'model.pt')
    row['checkpoint_sha256'] = sha256(folder / 'model.pt')
    write_json(folder / 'history.json', history); write_json(folder / 'metrics.json', row)
    write_json(folder / 'validation_probabilities.json', p.tolist())
    trials = read(Path(study) / 'trials.json'); trials.append(row); write_json(Path(study) / 'trials.json', trials)
    write_json(Path(study) / 'status.json', {"phase": "fitting", "completed_fits": len(trials)})
    print(f"{len(trials)}/65 {name}: dev {row['validation']['accuracy']:.2%}", flush=True)
    return row


def targets(pool, models, selector, seed):
    started = time.perf_counter()
    u = pool.x[pool.unlabeled]
    full = [ep.infer(models['full_' + side], u) for side in ('a', 'b')]
    fp = [ep.probabilities(r[0][0]) for r in full]
    full_q = ((fp[0] + fp[1]) / 2).astype(np.float32)
    full_w = ((fp[0].argmax(1) == fp[1].argmax(1)) & (fp[0].max(1) >= selector['confidence']) & (fp[1].max(1) >= selector['confidence'])).astype(np.float32)
    full_seconds = time.perf_counter() - started
    started = time.perf_counter()
    queries = [ep.infer(models['subset_' + side], u, views=True) for side in ('a', 'b')]
    # TeacherA is evaluated against known L_B; teacherB against known L_A.
    anchor_indices = [pool.teacher_b, pool.teacher_a]
    anchor_labels = [pool.labels_for(i) for i in anchor_indices]
    anchors = [ep.infer(models['subset_' + side], pool.x[i], views=True) for side, i in zip(('a', 'b'), anchor_indices)]
    q, profile_w, base, info = ep.select(*queries, *anchors, *anchor_labels, selector)
    _, embedding_w, _, embedding_info = ep.select(*queries, *anchors, *anchor_labels, selector, embedding=True)
    result = {"full_q": full_q, "selftrain_full_conf": full_w, "subset_q": q,
        "selftrain_subset_conf": base.astype(np.float32), "profile": profile_w, "embedding": embedding_w,
        "matched_conf": ep.matched_weights(profile_w, base, q, seed),
        "matched_random": ep.matched_weights(profile_w, base, q, seed, random=True),
        "subset_query_logits": np.stack([r[0] for r in queries]), "subset_query_embedding": np.stack([r[1] for r in queries]),
        "anchor_logits": np.stack([r[0] for r in anchors]), "anchor_embedding": np.stack([r[1] for r in anchors]),
        "anchor_y": np.stack(anchor_labels), "full_query_logits": np.stack([r[0] for r in full]),
        "u_rows": pool.unlabeled}
    elapsed = time.perf_counter() - started
    return result, {"full_target_seconds": full_seconds, "subset_target_and_selector_seconds": elapsed,
        "profile_anchor_support": info, "embedding_anchor_support": embedding_info,
        "admissions": {k: {"count": int(np.sum(result[k] > 0)), "weight_sum": float(result[k].sum())} for k in METHODS if k in result}}


def train(study):
    study = Path(study); config = frozen(study)
    if read(study / 'status.json')['phase'] != 'prepared' or read(study / 'trials.json'):
        raise ValueError("Preserve prior fitting; do not restart this study.")
    torch.set_num_threads(config['threads'])
    with np.load(study / 'selection.npz', allow_pickle=False) as d:
        vx, vy = d['x'].copy(), d['y'].copy()
    for seed in config['seeds']:
        pool = ep.LearnerPool.load(study / 'learner' / f'seed_{seed}.npz')
        models = {}
        for side, indices, offset in (('a', pool.teacher_a, 100), ('b', pool.teacher_b, 200)):
            for kind in ('full', 'subset'):
                name = f'teacher_{kind}_{side}_seed_{seed}'
                model, history, stats = ep.fit(pool.x[indices], pool.labels_for(indices), vx, vy, seed + offset, config['training'], subset=kind == 'subset')
                save_fit(study, name, seed, model, history, {**stats, "method": 'teacher_' + kind, "side": side}, vx, vy)
                models[kind + '_' + side] = model
        target, audit = targets(pool, models, config['selector'], seed)
        np.savez_compressed(study / f'targets_seed_{seed}.npz', **target)
        write_json(study / f'selector_seed_{seed}.json', audit)
        for method in METHODS:
            args = {}
            if method in target:
                args = {"ux": pool.x[pool.unlabeled], "target": target['full_q'] if method == 'selftrain_full_conf' else target['subset_q'], "weight": target[method]}
            model, history, stats = ep.fit(pool.x[pool.labeled], pool.y, vx, vy, seed + 300, config['training'],
                steps=config['training']['steps'] * (3 if method == 'supervised160_long' else 1), **args)
            save_fit(study, f'{method}_seed_{seed}', seed, model, history, {**stats, "method": method}, vx, vy)
    write_json(study / 'status.json', {"phase": "low_label_complete", "completed_fits": len(read(study / 'trials.json'))})


def reference(study):
    study = Path(study); config = frozen(study)
    if read(study / 'status.json') != {"phase": "low_label_complete", "completed_fits": 60}:
        raise ValueError("Finish and freeze all low-label fits before revealing reference405 labels.")
    torch.set_num_threads(config['threads'])
    with np.load(study / 'selection.npz', allow_pickle=False) as d:
        vx, vy = d['x'].copy(), d['y'].copy()
    with np.load(study / 'reference405.npz', allow_pickle=False) as d:
        x, y = d['x'].copy(), d['y'].copy()
    write_json(study / 'reference_label_access.json', {"created_utc": datetime.now(timezone.utc).isoformat(), "role": "405-label supervised reference only; all low-label targets and models already frozen."})
    for seed in config['seeds']:
        model, history, stats = ep.fit(x, y, vx, vy, seed + 300, config['training'])
        save_fit(study, f'supervised405_seed_{seed}', seed, model, history, {**stats, "method": 'supervised405'}, vx, vy)
    trials = read(study / 'trials.json')
    write_json(study / 'fit_freeze.json', {"created_utc": datetime.now(timezone.utc).isoformat(), "trials_sha256": sha256(study / 'trials.json'),
        "checkpoints": {r['folder']: r['checkpoint_sha256'] for r in trials},
        "targets": {f'targets_seed_{s}.npz': sha256(study / f'targets_seed_{s}.npz') for s in config['seeds']}})
    write_json(study / 'status.json', {"phase": "fits_complete", "completed_fits": len(trials)})


def evaluate(study):
    study = Path(study); config = frozen(study, endpoint=True)
    if read(study / 'status.json') != {"phase": "fits_complete", "completed_fits": 65}:
        raise ValueError("Complete all65 fits before retrospective endpoint/oracle diagnostics.")
    if (study / 'endpoint_access.json').exists():
        raise ValueError("Endpoint accessed already; preserve outputs and use verify only.")
    f = read(study / 'fit_freeze.json')
    if sha256(study / 'trials.json') != f['trials_sha256'] or any(sha256(study / k / 'model.pt') != v for k, v in f['checkpoints'].items()) or any(sha256(study / k) != v for k, v in f['targets'].items()):
        raise ValueError("A candidate/target changed after fitting.")
    write_json(study / 'endpoint_access.json', {"state": "started", "created_utc": datetime.now(timezone.utc).isoformat(), "scope": config['scope']})
    out = study / 'endpoint'; out.mkdir(exist_ok=False)
    try:
        torch.set_num_threads(config['threads'])
        with np.load(ENDPOINT / 'evaluation/features.npz', allow_pickle=False) as a:
            x, y = a['camera_motion'].copy(), a['y'].copy()
        rows = read(ENDPOINT / 'evaluation/manifest.json')['records']
        np.savez_compressed(out / 'features.npz', x=x, y=y)
        write_json(out / 'manifest.json', {"records": rows, "scope": config['scope']})
        results = []
        for trial in read(study / 'trials.json'):
            if trial['method'].startswith('teacher_'):
                continue
            model, _ = load_model(study / trial['folder'] / 'model.pt')
            logits, _ = ep.infer(model, x); p = ep.probabilities(logits[0])
            r = {"folder": trial['folder'], "method": trial['method'], "seed": trial['seed'], "pooled": old.motion.previous.fitting.metric_set(p, y),
                "sittings": {str(s): old.motion.previous.fitting.metric_set(p[[i for i, row in enumerate(rows) if row['sitting'] == s]], y[[i for i, row in enumerate(rows) if row['sitting'] == s]]) for s in (1, 2)}}
            write_json(out / (trial['folder'] + '_probabilities.json'), p.tolist()); results.append(r)
        write_json(out / 'results.json', results)
        write_json(study / 'endpoint_access.json', {"state": "complete", "created_utc": datetime.now(timezone.utc).isoformat(), "models": len(results), "clips": len(y), "scope": config['scope']})
        write_json(study / 'status.json', {"phase": "complete", "completed_fits": 65})
    except BaseException as e:
        write_json(study / 'endpoint_access.json', {"state": "failed_inspected", "error": str(e)}); raise
    print(f'Retrospective endpoint saved for{len(results)} students; no model activated.')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('prepare', 'train', 'reference', 'evaluate'))
    p.add_argument('study', nargs='?', type=Path)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        if args.study is None: p.error('Provide the frozen study path.')
        {'train': train, 'reference': reference, 'evaluate': evaluate}[args.action](args.study)


if __name__ == '__main__':
    main()
