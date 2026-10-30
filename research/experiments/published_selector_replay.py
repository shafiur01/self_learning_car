"""017: static published selector components; no torch, training or inference.

This is a retrospective diagnostic, NOT SoftMatch/FreeMatch reproduction.
Run with: .venv/bin/python -m research.experiments.published_selector_replay
"""

import csv
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
SAVED = PROJECT / 'research/experiments/runs/evidence_profile_diagnostic/20261016T142255449351Z'
PARENT = PROJECT / 'research/experiments/runs/evidence_profile_budget160/20261016T133936682213Z'
PLAN = PROJECT / 'research/findings/017_literature_grounded_decision.md'
TEST = PROJECT / 'research/experiments/test_published_selector_replay.py'
SEEDS = (11, 22, 33, 44, 55)
CLASSES = ('FORWARD', 'BACKWARD', 'LEFT', 'RIGHT', 'STOP', 'NO_COMMAND', 'UP', 'DOWN')
METHODS = ('original_dual95', 'ensemble95', 'global_snapshot', 'class_snapshot',
           'gaussian_snapshot', 'gaussian_ua_snapshot')
MOVEMENT = (0, 1, 2, 3, 6, 7)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def probabilities(q):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 8 or len(q) < 2:
        raise ValueError('Require at least two eight-class probability vectors.')
    if not np.isfinite(q).all() or np.any(q < 0) or np.any(q > 1):
        raise ValueError('Invalid probability values.')
    if not np.allclose(q.sum(1), 1, rtol=0, atol=1e-6):
        raise ValueError('Probability rows must sum to one.')
    return q / q.sum(1, keepdims=True)


def gaussian(confidence, mean, variance):
    # SoftMatch Section 4.1: unbiased confidence variance / 4 (2 sigma).
    below = np.minimum(np.asarray(confidence) - mean, 0.0)
    # If variance is zero, use the continuous limit: below-mean gets no mass.
    tolerance = 8 * np.finfo(np.float64).eps
    if variance <= tolerance ** 2:
        return (below >= -tolerance).astype(np.float64)
    return np.exp(-(below ** 2) / (2 * variance / 4))


def components(q):
    """Prediction-only static substitution for online EMA statistics.

    UA changes confidence used for weights, NEVER the pseudo-label target.
    No ground-truth, student predictions or evaluation inputs enter this API.
    """
    q = probabilities(q)
    target = q.argmax(1)
    confidence = q.max(1)
    mean = float(confidence.mean())
    variance = float(confidence.var(ddof=1))
    marginal = q.mean(0)
    threshold = mean * marginal / marginal.max()
    aligned = q / np.maximum(marginal, np.finfo(np.float64).tiny)
    aligned /= aligned.sum(1, keepdims=True)
    weights = {
        'ensemble95': (confidence >= .95).astype(np.float64),
        'global_snapshot': (confidence >= mean).astype(np.float64),
        'class_snapshot': (confidence >= threshold[target]).astype(np.float64),
        'gaussian_snapshot': gaussian(confidence, mean, variance),
        'gaussian_ua_snapshot': gaussian(aligned.max(1), mean, variance),
    }
    for w in weights.values():
        if not (np.isfinite(w).all() and ((w >= 0) & (w <= 1)).all()):
            raise ValueError('Weights must be finite in [0,1].')
    return target, weights, {'mean_confidence': mean, 'unbiased_variance': variance,
                             'marginal': marginal.tolist(), 'thresholds': threshold.tolist()}


def checked_load(path, allowed, ledger, phase):
    resolved = Path(path).resolve()
    if resolved not in allowed:
        raise ValueError(f'Forbidden input: {resolved}')
    ledger.append({'phase': phase, 'file': str(resolved), 'utc': utc()})
    with np.load(resolved, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def stats(weight, actual, target, student):
    total = float(weight.sum())
    wrong = float(weight[actual != target].sum())
    useful_proxy = (actual == target) & (student != actual)
    false_move = (actual == 5) & np.isin(target, MOVEMENT)
    none_mass = float(weight[actual == 5].sum())
    return {'pairs': len(weight), 'positive_weight_pairs': int((weight > 0).sum()),
            'mass': total, 'wrong_mass': wrong,
            'wrong_fraction': wrong / total if total else None,
            'correct_student_error_mass': float(weight[useful_proxy].sum()),
            'none_pairs': int((actual == 5).sum()), 'none_mass': none_mass,
            'none_to_movement_mass': float(weight[false_move].sum()),
            'none_to_movement_fraction': float(weight[false_move].sum()) / none_mass if none_mass else None}


def table(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.perf_counter()
    study = PROJECT / 'research/experiments/runs/published_selector_replay' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (study / 'weights').mkdir(parents=True, exist_ok=False)
    score_paths = [SAVED / f'scores/seed_{s}.npz' for s in SEEDS]
    oracle_path = PARENT / 'oracle.npz'
    inputs = [*score_paths, oracle_path]
    sources = [Path(__file__), TEST, PLAN]
    for source in sources:
        shutil.copyfile(source, study / source.name)
    freeze = {'created_utc': utc(), 'seeds': SEEDS, 'methods': METHODS,
              'input_sha256': {str(p): digest(p) for p in inputs},
              'source_sha256': {p.name: digest(p) for p in sources},
              'model_fits': 0, 'forward_passes': 0,
              'scope': 'Retrospective static component diagnostic; no full algorithm reproduction.'}
    write(study / 'freeze.json', freeze)
    write(study / 'status.json', {'phase': 'prepared'})
    ledger, parameters = [], []
    # Score phase: only saved hidden-training probabilities, masks and row IDs.
    for seed, path in zip(SEEDS, score_paths):
        d = checked_load(path, set(score_paths), ledger, 'score')
        target, weights, param = components(d['q'])
        np.testing.assert_array_equal(target, d['pred'])
        original = d['agree'] & (d['confidence'] >= .95)
        np.testing.assert_array_equal(original, d['base'])
        weights['original_dual95'] = original.astype(np.float64)
        # student_p was already saved in 015; it is NOT passed to components().
        np.savez_compressed(study / f'weights/seed_{seed}.npz',
                            u_rows=d['u_rows'], target=target,
                            student=probabilities(d['student_p']).argmax(1), **weights)
        parameters.append({'seed': seed, **param})
    write(study / 'parameters.json', parameters)
    weight_paths = sorted((study / 'weights').glob('*.npz'))
    weight_freeze = {'created_utc': utc(), 'sha256': {str(p): digest(p) for p in weight_paths}}
    write(study / 'weight_freeze.json', weight_freeze)
    write(study / 'status.json', {'phase': 'weights_frozen'})
    # Separate oracle phase. These training labels have been inspected before.
    write(study / 'oracle_access.json', {'created_utc': utc(), 'retrospective': True})
    oracle = checked_load(oracle_path, {oracle_path}, ledger, 'oracle')['y']
    per_seed, grouped = [], []
    pooled = {method: [] for method in METHODS}
    for seed, path in zip(SEEDS, weight_paths):
        d = checked_load(path, set(weight_paths), ledger, 'analysis')
        actual = oracle[d['u_rows']]
        assert actual.shape == d['target'].shape == (245,)
        assert np.isin(actual, np.arange(8)).all()
        for method in METHODS:
            w = d[method]
            row = {'seed': seed, 'method': method, **stats(w, actual, d['target'], d['student'])}
            per_seed.append(row)
            pooled[method].append((w, actual, d['target'], d['student']))
            for axis, labels in [('actual', actual), ('predicted', d['target'])]:
                for c, name in enumerate(CLASSES):
                    mask = labels == c
                    grouped.append({'seed': seed, 'method': method, 'axis': axis,
                                    'class': name, **stats(w[mask], actual[mask],
                                                         d['target'][mask], d['student'][mask])})
    totals = []
    for method, batches in pooled.items():
        joined = [np.concatenate([b[i] for b in batches]) for i in range(4)]
        totals.append({'method': method, **stats(*joined)})
    table(study / 'per_seed.csv', per_seed)
    table(study / 'per_class.csv', grouped)
    write(study / 'summary.json', totals)
    write(study / 'access.json', ledger)
    # Independent aggregate accounting identities, not a model-performance test.
    for summary in totals:
        for metric in ('mass', 'wrong_mass', 'correct_student_error_mass', 'none_to_movement_mass'):
            np.testing.assert_allclose(summary[metric], sum(r[metric] for r in per_seed if r['method'] == summary['method']))
            np.testing.assert_allclose(summary[metric], sum(r[metric] for r in grouped if r['method'] == summary['method'] and r['axis'] == 'actual'))
    for p, h in freeze['input_sha256'].items():
        assert digest(p) == h, f'Parent input changed: {p}'
    for p, h in weight_freeze['sha256'].items():
        assert digest(p) == h, f'Weight changed after freeze: {p}'
    assert weight_freeze['created_utc'] < json.loads((study / 'oracle_access.json').read_text())['created_utc']
    write(study / 'verification.json', {'parent_inputs_unchanged': True,
          'all_weights_frozen_before_new_oracle_access': True, 'original_masks_reproduced': 5,
          'aggregate_accounting_checks': True, 'model_fits': 0, 'forward_passes': 0,
          'selection_or_endpoint_payload_reads': 0, 'pairs': 1225,
          'elapsed_seconds': time.perf_counter() - started})
    write(study / 'status.json', {'phase': 'complete'})
    manifest = {str(p.relative_to(study)): digest(p) for p in study.rglob('*') if p.is_file()}
    write(study / 'artifact_sha256.json', manifest)
    print(study.relative_to(PROJECT))
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
