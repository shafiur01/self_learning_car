"""Verify pilot013 archives and report every result after choices are frozen."""

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import evidence_profile as ep
from . import evidence_profile_run as run
from .run import sha256, write_json


def verify(study, raw=False):
    study = Path(study); config = run.frozen(study, endpoint=True)
    torch.set_num_threads(config['threads'])
    trials = run.read(study / 'trials.json')
    if len(trials) != 65 or run.read(study / 'status.json')['phase'] != 'complete':
        raise ValueError('Require complete65 fits and retrospective endpoint.')
    fit_freeze = run.read(study / 'fit_freeze.json')
    if sha256(study / 'trials.json') != fit_freeze['trials_sha256']:
        raise ValueError('Frozen trials changed.')
    for name, digest in fit_freeze['targets'].items():
        if sha256(study / name) != digest: raise ValueError('Frozen pseudo-target changed.')
    with np.load(study / 'selection.npz', allow_pickle=False) as a:
        vx, vy = a['x'].copy(), a['y'].copy()
    vectors = 0; initial = {}; batches = {}; teacher_initial = {}; teacher_batches = {}
    for trial in trials:
        folder = study / trial['folder']
        if sha256(folder / 'model.pt') != trial['checkpoint_sha256'] or trial['checkpoint_sha256'] != fit_freeze['checkpoints'][trial['folder']]:
            raise ValueError('Checkpoint changed.')
        model, c = run.load_model(folder / 'model.pt')
        if any(c[k] != v for k, v in {'seed': trial['seed'], 'folder': trial['folder'], 'best_step': trial['best_step'],
            'config_sha256': sha256(study / 'config.json'), 'freeze_sha256': sha256(study / 'freeze.json')}.items()):
            raise ValueError('Checkpoint provenance changed.')
        p = ep.probabilities(ep.infer(model, vx)[0][0])
        np.testing.assert_allclose(p, run.read(folder / 'validation_probabilities.json'), rtol=1e-6, atol=1e-7)
        if run.old.motion.previous.fitting.metric_set(p, vy) != trial['validation'] or ep.state_hash(model) != trial['state_sha256'] or run.read(folder / 'metrics.json') != trial:
            raise ValueError('Validation/state/metrics differ.')
        history = run.read(folder / 'history.json')
        selected = min(history, key=lambda r: r['selection_loss'])
        if selected['step'] != trial['best_step'] or len(history) != 28 or [r['step'] for r in history] != list(range(trial['optimizer_steps']//28, trial['optimizer_steps']+1, trial['optimizer_steps']//28)):
            raise ValueError('Checkpoint minimum/check schedule differs.')
        if not np.isclose(selected['selection_loss'], trial['validation']['nll'], rtol=1e-6, atol=1e-7):
            raise ValueError('Minimum selection loss does not match saved model.')
        if trial['method'].startswith('teacher_'):
            key = (trial['seed'], trial['side'])
            teacher_initial.setdefault(key, set()).add(trial['initial_state_sha256'])
            teacher_batches.setdefault(key, set()).add(trial['batch_indices_sha256'])
            if trial['training_class_counts'] != [5]*8: raise ValueError('Teacher lacks five labels/class.')
        else:
            initial.setdefault(trial['seed'], set()).add(trial['initial_state_sha256'])
            if trial['method'] not in ('supervised80_long', 'supervised405'):
                batches.setdefault(trial['seed'], set()).add(trial['batch_indices_sha256'])
            if trial['method'] != 'supervised405' and trial['training_class_counts'] != [10]*8:
                raise ValueError('Sparse student lacks ten labels/class.')
        vectors += len(vy)
    if any(len(v) != 1 for group in (initial, batches, teacher_initial, teacher_batches) for v in group.values()):
        raise ValueError('Declared initialization or labeled-batch pairing broken.')
    # Recreate all selectors from the same frozen teachers, without opening oracle labels.
    for seed in config['seeds']:
        pool = ep.LearnerPool.load(study / 'learner' / f'seed_{seed}.npz')
        models = {f'{kind}_{side}': run.load_model(study / f'teacher_{kind}_{side}_seed_{seed}' / 'model.pt')[0] for kind in ('full', 'subset') for side in ('a', 'b')}
        expected, _ = run.targets(pool, models, config['selector'], seed)
        with np.load(study / f'targets_seed_{seed}.npz', allow_pickle=False) as a:
            if set(a.files) != set(expected): raise ValueError('Target fields differ.')
            for k, value in expected.items(): np.testing.assert_allclose(a[k], value, rtol=1e-6, atol=1e-7)
    results = run.read(study / 'endpoint/results.json')
    with np.load(study / 'endpoint/features.npz', allow_pickle=False) as a:
        x, y = a['x'].copy(), a['y'].copy()
    rows = run.read(study / 'endpoint/manifest.json')['records']
    if len(results) != 45: raise ValueError('Missing student endpoint.')
    for r in results:
        model, _ = run.load_model(study / r['folder'] / 'model.pt')
        p = ep.probabilities(ep.infer(model, x)[0][0])
        np.testing.assert_allclose(p, run.read(study / 'endpoint' / (r['folder']+'_probabilities.json')), rtol=1e-6, atol=1e-7)
        if run.old.motion.previous.fitting.metric_set(p, y) != r['pooled']: raise ValueError('Endpoint metric changed.')
        for s in (1, 2):
            i = [j for j, row in enumerate(rows) if row['sitting'] == s]
            if run.old.motion.previous.fitting.metric_set(p[i], y[i]) != r['sittings'][str(s)]: raise ValueError('Sitting metric changed.')
        vectors += len(y)
    count = 0
    if raw:
        registry = run.read(study / 'registry.json')
        source_manifest = run.read(run.SOURCE / 'manifest.json')
        source_rows = {r['file']: r for r in source_manifest['records']}
        pool = ep.LearnerPool.load(study / 'learner/seed_11.npz')
        for records, arrays in ((registry['records'], pool.x), ([source_rows[f] for f in registry['selection_files']], vx), (rows, x)):
            for row, expected in zip(records, arrays):
                xyz, times, visible, meta = run.old.motion.previous.training.load_clip(row['file'], config['contract'])
                actual, _ = run.old.motion.rep.features(xyz, times, visible, meta['aspect'], 'camera_motion')
                np.testing.assert_array_equal(actual, expected); count += 1
    result = {'checkpoints_verified': 65, 'prediction_vectors_verified': vectors, 'selector_seed_bundles_reproduced': 5,
        'raw_feature_arrays_reextracted': count, 'pairing_verified': True, 'scope': config['scope']}
    write_json(study / 'verification.json', result)
    return result


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def report(study):
    study = Path(study); config = run.frozen(study, endpoint=True)
    if run.read(study / 'status.json')['phase'] != 'complete': raise ValueError('Complete all models before analysis.')
    output = study / 'analysis'; output.mkdir(exist_ok=False)
    # This access happens after all models/targets froze, never feeds back to fitting.
    write_json(study / 'oracle_access.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'role': 'Post-freeze diagnostics only; no cutoff/model decisions.'})
    with np.load(study / 'oracle.npz', allow_pickle=False) as a: truth = a['y'].copy()
    results, trials = run.read(study / 'endpoint/results.json'), run.read(study / 'trials.json')
    metrics = ('accuracy', 'macro_f1', 'nll', 'no_command_to_motion_rate', 'stop_recall')
    groups = {}; model_rows = []; classes = []; diagnostics = []; error_rows = []
    with np.load(study / 'endpoint/features.npz', allow_pickle=False) as a: endpoint_y = a['y'].copy()
    endpoint_rows = run.read(study / 'endpoint/manifest.json')['records']
    for method in (*run.METHODS, 'supervised405'):
        members = [r for r in results if r['method'] == method]
        groups[method] = {}
        for scope in ('pooled', '1', '2'):
            measured = [r['pooled'] if scope == 'pooled' else r['sittings'][scope] for r in members]
            groups[method][scope] = {k: {'mean': float(np.mean([r[k] for r in measured])), 'seed_sd': float(np.std([r[k] for r in measured], ddof=1))} for k in metrics}
        for r in members:
            model_rows.append({'method': method, 'seed': r['seed'], **{k: r['pooled'][k] for k in metrics},
                'sitting1_accuracy': r['sittings']['1']['accuracy'], 'sitting2_accuracy': r['sittings']['2']['accuracy']})
            for label, values in r['pooled']['per_class'].items(): classes.append({'method': method, 'seed': r['seed'], 'label': label, **values})
            p = np.array(run.read(study / 'endpoint' / (r['folder']+'_probabilities.json')))
            for i in np.flatnonzero(p.argmax(1) != endpoint_y):
                error_rows.append({'method': method, 'seed': r['seed'], 'file': endpoint_rows[i]['file'], 'sitting': endpoint_rows[i]['sitting'],
                    'actual': run.LABELS[int(endpoint_y[i])], 'predicted': run.LABELS[int(p[i].argmax())], 'raw_score': float(p[i].max())})
    for seed in config['seeds']:
        pool = ep.LearnerPool.load(study / 'learner' / f'seed_{seed}.npz')
        with np.load(study / f'targets_seed_{seed}.npz', allow_pickle=False) as t:
            for method in run.METHODS:
                if method not in t.files: continue
                w = t[method].copy(); q = t['full_q'] if method == 'selftrain_full_conf' else t['subset_q']
                errors = (q.argmax(1) != truth[pool.unlabeled]).astype(float)
                eligible = t['selftrain_full_conf'] > 0 if method == 'selftrain_full_conf' else t['selftrain_subset_conf'] > 0
                mass = float(w.sum()); count = int(np.sum(w > 0))
                covariance = float(np.mean((w[eligible]-w[eligible].mean()) * (errors[eligible]-errors[eligible].mean()))) if eligible.any() else None
                diagnostics.append({'seed': seed, 'method': method, 'admitted': count, 'weight_sum': mass, 'coverage_of_325': count/325,
                    'unweighted_error': float(errors[w>0].mean()) if count else None, 'weighted_error': float(np.sum(w*errors)/mass) if mass else None,
                    'base_confidence_error': float(errors[eligible].mean()) if eligible.any() else None, 'weight_error_covariance_on_base': covariance})
    paired = {}
    for baseline in ('selftrain_subset_conf', 'matched_conf', 'matched_random', 'embedding', 'supervised80', 'supervised80_long', 'supervised405'):
        a = {r['seed']: r for r in results if r['method'] == 'profile'}
        b = {r['seed']: r for r in results if r['method'] == baseline}
        paired[baseline] = {}
        for scope in ('pooled', '1', '2'):
            value = lambda r: (r['pooled'] if scope == 'pooled' else r['sittings'][scope])['macro_f1']
            d = [value(a[s])-value(b[s]) for s in config['seeds']]
            paired[baseline][scope] = {'differences': d, 'mean': float(np.mean(d)), 'positive_pairs': sum(x>0 for x in d)}
    costs = []
    for trial in trials:
        if trial['method'].startswith('teacher_'): continue
        teachers = [t for t in trials if t['seed'] == trial['seed'] and t['method'] == ('teacher_full' if trial['method'] == 'selftrain_full_conf' else 'teacher_subset')]
        ssl = trial['method'] not in ('supervised80', 'supervised80_long', 'supervised405')
        audit = run.read(study / f"selector_seed_{trial['seed']}.json")
        costs.append({'method': trial['method'], 'seed': trial['seed'], 'student_fit_seconds': trial['fit_seconds'],
            'charged_teacher_fit_seconds': sum(t['fit_seconds'] for t in teachers) if ssl else 0,
            'charged_target_selector_envelope_seconds': audit['full_target_seconds' if trial['method']=='selftrain_full_conf' else 'subset_target_and_selector_seconds'] if ssl else 0,
            'optimizer_steps_charged': trial['optimizer_steps'] + (sum(t['optimizer_steps'] for t in teachers) if ssl else 0)})
    summary = {'groups': groups, 'profile_minus_baseline_f1': paired, 'pseudo_label_diagnostics': diagnostics,
        'physical_fits': 65, 'physical_fit_seconds': float(sum(t['fit_seconds'] for t in trials)), 'new_labels_collected': 0, 'scope': config['scope']}
    write_json(output / 'summary.json', summary)
    write_csv(output / 'model_results.csv', model_rows); write_csv(output / 'per_class_results.csv', classes)
    write_csv(output / 'pseudo_label_diagnostics.csv', diagnostics); write_csv(output / 'costs.csv', costs)
    if error_rows: write_csv(output / 'endpoint_errors.csv', error_rows)
    lines = ['# Pilot013: evidence-profile self-training', '', config['scope'], '',
        '405 original training inputs;80 revealed labels +325 label-hidden inputs.48 development labels select checkpoints;96 historically inspected endpoint clips evaluated only after all candidates froze. All eight commands have10 revealed labels/seed; no validation clips were added to training.', '',
        '| Method | Endpoint accuracy (%) | MacroF1 | Sitting1 /2 accuracy (%) | NONE→motion (%) | STOP recall (%) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for method, g in groups.items():
        m = g['pooled']; lines.append(f"| {method} | {m['accuracy']['mean']*100:.2f} | {m['macro_f1']['mean']:.4f} | {g['1']['accuracy']['mean']*100:.2f} / {g['2']['accuracy']['mean']*100:.2f} | {m['no_command_to_motion_rate']['mean']*100:.2f} | {m['stop_recall']['mean']*100:.2f} |")
    primary = paired['selftrain_subset_conf']['pooled']
    lines += ['', f"Primary paired profile−same-subset-teacher-confidence macroF1: {primary['mean']:+.6f}, positive in{primary['positive_pairs']}/5. Seed variation is not population uncertainty. All outcomes retained.", '',
        '| Selector | Mean admitted /325 | Mean weight mass | Mean weighted pseudo-error (supported seeds only) |', '| --- | ---: | ---: | ---: |']
    for method in ('selftrain_full_conf', 'selftrain_subset_conf', 'profile', 'embedding', 'matched_conf', 'matched_random'):
        d = [x for x in diagnostics if x['method']==method]; e = [x['weighted_error'] for x in d if x['weighted_error'] is not None]
        error = f'{np.mean(e)*100:.2f}%' if e else 'undefined: zero coverage'
        lines.append(f"| {method} | {np.mean([x['admitted'] for x in d]):.1f} | {np.mean([x['weight_sum'] for x in d]):.2f} | {error} |")
    lines += ['', f"65 fresh fits, numerical fit loops{summary['physical_fit_seconds']:.2f}s total. Method costs charge both teachers even when reused physically. Supervised80_long matches1680 optimizer updates, not exact FLOPs/runtime or selection exposure. Common selector envelope includes unused views/embeddings; no exact efficiency guarantee.", '',
        'No new personal recordings, dependency or camera activation. No actual annotation saving, novel algorithm/SOTA, unseen-user reliability or robot action outcome established. This is one budget and a private retrospective single-user pilot; stronger external baselines/benchmarks are still required.', '',
        'Artifacts: all65 weights/histories/development probabilities, frozen per-seed target tensors/anchor support,45 endpoint score sets, complete class/error tables, post-freeze oracle diagnostics, costs, source/config/input fingerprints and verification. See the frozen protocol for all fixed choices.']
    (study / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    plot(output, summary, results)
    return summary


def plot(output, summary, results):
    os.environ.setdefault('MPLCONFIGDIR', str(output / 'plot_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = {'supervised80': 'Supervised80', 'supervised80_long': 'Supervised80,1680 steps', 'selftrain_full_conf': 'Full teachers + confidence',
        'selftrain_subset_conf': 'Subset teachers + confidence', 'profile': 'Response profile', 'embedding': 'Embedding support', 'matched_conf': 'Count-matched confidence',
        'matched_random': 'Count-matched random', 'supervised405': 'Supervised405 reference'}
    methods = list(summary['groups']); groups = summary['groups']; fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    colors = ['#216a87' if m=='profile' else '#9cb2bd' for m in methods]
    a = axes[0, 0]; values = [groups[m]['pooled']['accuracy']['mean']*100 for m in methods]
    a.barh(range(len(methods)), values, color=colors); a.invert_yaxis()
    a.set(yticks=range(len(methods)), yticklabels=[names[m] for m in methods], xlim=(0, 105), xlabel='Accuracy (%)', title='A  Retrospective endpoint96 · allfive seeds')
    for i, v in enumerate(values): a.text(v+.6, i, f'{v:.2f}', va='center', fontsize=9)
    a = axes[0, 1]; baselines = ['selftrain_subset_conf', 'matched_conf', 'embedding', 'supervised80_long']
    for i, m in enumerate(baselines):
        d = np.array(summary['profile_minus_baseline_f1'][m]['pooled']['differences'])*100
        a.bar(i, d.mean(), color='#216a87' if d.mean()>0 else '#bb7d87'); a.scatter(i+np.linspace(-.12,.12,5), d, color='#19333f', zorder=3)
    a.axhline(0, color='gray', lw=1)
    a.set(xticks=range(4), xticklabels=['Subset\nconfidence', 'Matched\nconfidence', 'Embedding', 'Long\nsupervised'], ylabel='Profile minus baseline macroF1 ×100', title='B  Paired differences; negatives retained')
    a = axes[1, 0]; selectors = ['selftrain_full_conf','selftrain_subset_conf','profile','embedding','matched_conf','matched_random']
    counts = [np.mean([d['admitted'] for d in summary['pseudo_label_diagnostics'] if d['method']==m]) for m in selectors]
    a.bar(range(6), counts, color=['#216a87' if m=='profile' else '#9cb2bd' for m in selectors])
    a.set(xticks=range(6), xticklabels=['Full\nconfidence','Subset\nconfidence','Profile','Embedding','Matched\nconfidence','Matched\nrandom'], ylim=(0, 325), ylabel='Mean admitted clips /325', title='C  Coverage is reported alongside target quality')
    for i, v in enumerate(counts): a.text(i, v+4, f'{v:.1f}', ha='center')
    a = axes[1, 1]; matrix = np.array([[np.mean([r['pooled']['per_class'][label]['recall'] for r in results if r['method']==m]) for label in run.LABELS] for m in methods])
    a.imshow(matrix, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    a.set(xticks=range(8), xticklabels=['F','B','L','R','STOP','NONE','UP','DOWN'], yticks=range(9), yticklabels=[names[m] for m in methods], title='D  Per-command recall (mean %)')
    for i in range(9):
        for j in range(8): a.text(j, i, f'{matrix[i,j]*100:.0f}', ha='center', va='center', color='white' if matrix[i,j]>.65 else '#18313e', fontsize=8)
    fig.suptitle('Pilot013: trust model-generated labels using evidence responses?', fontsize=16)
    fig.supxlabel('80 training labels +48 development labels ·405 training inputs · inspected endpoint96 · no population or label-saving claim', fontsize=10)
    fig.savefig(output / 'figure.png', dpi=160, bbox_inches='tight'); fig.savefig(output / 'figure.pdf', bbox_inches='tight'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('verify','report')); p.add_argument('study', type=Path); p.add_argument('--raw', action='store_true')
    args = p.parse_args(); print(json.dumps(verify(args.study, args.raw) if args.action=='verify' else report(args.study), indent=2))


if __name__ == '__main__': main()
