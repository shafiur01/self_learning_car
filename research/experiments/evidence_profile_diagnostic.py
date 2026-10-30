"""Frozen-model diagnostic015; no fitting and no endpoint feature/score access."""

import argparse
import csv
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import evidence_profile as ep
from . import evidence_profile_budget_run as parent
from .run import PROJECT, sha256, write_json

ID = '015_evidence_profile_diagnostic'
PARENT = PROJECT / 'research/experiments/runs/evidence_profile_budget160/20261016T133936682213Z'
OUTPUT = PROJECT / 'research/experiments/runs/evidence_profile_diagnostic'
SOURCES = tuple(dict.fromkeys((*parent.SOURCES,
    'research/experiments/evidence_profile_diagnostic.py',
    'research/experiments/test_evidence_profile_diagnostic.py',
    'research/EVIDENCE_PROFILE_DIAGNOSTIC_PROTOCOL.md')))
METHODS = ('confidence', 'profile', 'embedding', 'random')
STAGES = ('disagreement', 'low_confidence', 'unsupported_anchors', 'outside_radius', 'accepted')
BINS = np.array([.5, .7, .85, .95])
FRACTIONS = (.10, .25, .50, 1.0)


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def dense_support(query, bank, y, p, proposed, config):
    """Original anchor rule, retaining distances outside its acceptance radius."""
    correct_bank = bank[np.arange(len(bank)), y]
    center = np.median(correct_bank, axis=0)
    scale = np.maximum(config['scale_floor'], 1.4826 * np.median(np.abs(correct_bank - center), axis=0))
    ratios = np.full(len(query), np.inf)
    inside = np.zeros(len(query), dtype=bool)
    supported = np.zeros(len(query), dtype=bool)
    info = []
    for c in range(8):
        anchors = np.flatnonzero((y == c) & (p.argmax(1) == c) & (p[:, c] >= config['anchor_confidence']))
        row = {'class': c, 'eligible_anchors': len(anchors), 'supported': len(anchors) >= config['minimum_anchors']}
        if row['supported']:
            b = bank[anchors, c] / scale
            d = ((b[:, None] - b[None]) ** 2).sum(2)
            np.fill_diagonal(d, np.inf)
            radius = max(config['radius_floor_squared'], float(np.median(d.min(1))))
            indices = np.flatnonzero(proposed == c)
            nearest = ((query[indices, c, None, :] / scale - b[None]) ** 2).sum(2).min(1) if len(indices) else np.array([])
            ratios[indices] = nearest.astype(np.float64) / radius
            inside[indices] = nearest <= radius
            supported[indices] = True
            row['radius_squared'] = radius
        info.append(row)
    return ratios, inside, supported, info


def candidate_scores(targets, config):
    """Only predictions and revealed anchor labels enter this interface."""
    pa, pb = ep.probabilities(targets['subset_query_logits'][:, 0])
    pred = targets['subset_q'].argmax(1)
    agree = pa.argmax(1) == pb.argmax(1)
    confidence = np.minimum(pa[np.arange(len(pred)), pred], pb[np.arange(len(pred)), pred])
    base = agree & (confidence >= config['confidence'])
    out = {'u_rows': targets['u_rows'], 'pred': pred, 'teacher_a': pa.argmax(1),
           'teacher_b': pb.argmax(1), 'confidence': confidence, 'agree': agree,
           'base': base, 'q': targets['subset_q']}
    info = {}
    for name in ('profile', 'embedding'):
        distances, within, supports, details = [], [], [], []
        for side in range(2):
            logits, anchor_logits = targets['subset_query_logits'][side], targets['anchor_logits'][side]
            if name == 'profile':
                query, bank = ep.responses(logits), ep.responses(anchor_logits)
            else:
                query = np.repeat(targets['subset_query_embedding'][side, :, None, :], 8, axis=1)
                bank = np.repeat(targets['anchor_embedding'][side, :, None, :], 8, axis=1)
            d, inside, supported, detail = dense_support(query, bank, targets['anchor_y'][side],
                                                       ep.probabilities(anchor_logits[0]), pred, config)
            distances.append(d); within.append(inside); supports.append(supported); details.append(detail)
        maximum = np.maximum(*distances)
        supported = np.logical_and(*supports)
        inside = np.logical_and(*within)
        weights = np.where(base & inside, np.exp(-maximum), 0).astype(np.float32)
        np.testing.assert_allclose(weights, targets[name], rtol=2e-6, atol=2e-7)
        out.update({name + '_distance': maximum, name + '_supported': supported,
                    name + '_inside': inside, name + '_accepted': weights > 0})
        info[name] = details
    np.testing.assert_array_equal(base, targets['selftrain_subset_conf'] > 0)
    out['shared_support'] = out['profile_supported'] & out['embedding_supported']
    return out, info


def first_failure(agree, confidence, support, inside):
    stage = np.zeros(len(agree), dtype=np.int64)
    stage[agree] = 1
    stage[confidence] = 2
    stage[confidence & support] = 3
    stage[confidence & support & inside] = 4
    return stage


def matched_pick(score, predicted, domain, fraction, bins=None):
    """Equal cell quotas for every ranker, deterministic row-order tie breaks."""
    if not 0 < fraction <= 1:
        raise ValueError('Coverage fraction must be in (0,1].')
    if not np.isfinite(score[domain]).all():
        raise ValueError('Rank all methods on the same supported domain.')
    cells = predicted if bins is None else predicted * 5 + bins
    chosen = []
    for cell in sorted(set(cells[domain])):
        indices = np.flatnonzero(domain & (cells == cell))
        ordered = indices[np.lexsort((indices, score[indices]))]
        chosen.extend(ordered[:math.ceil(fraction * len(indices))].tolist())
    return np.array(sorted(chosen), dtype=np.int64)


def selections(scores, seed):
    n = len(scores['pred'])
    random_order = np.empty(n, dtype=np.int64)
    random_order[np.random.default_rng(seed + 15_000_000).permutation(n)] = np.arange(n)
    rankers = {'confidence': -scores['confidence'], 'profile': scores['profile_distance'],
               'embedding': scores['embedding_distance'], 'random': random_order}
    primary = scores['agree'] & scores['shared_support']
    domains = {'class_quota': primary, 'frozen_gate': primary & scores['base'],
               'confidence_matched': primary}
    bins = np.digitize(scores['confidence'], BINS)
    rows = []
    for domain, mask in domains.items():
        for fraction in ((.25,) if domain == 'confidence_matched' else FRACTIONS):
            counts = []
            for method, score in rankers.items():
                chosen = matched_pick(score, scores['pred'], mask, fraction,
                                      bins if domain == 'confidence_matched' else None)
                cells = scores['pred'] * 5 + bins if domain == 'confidence_matched' else scores['pred']
                counts.append(np.bincount(cells[chosen], minlength=40))
                rows.append({'domain': domain, 'fraction': fraction, 'method': method,
                             'available': int(mask.sum()), 'rows': chosen.tolist()})
            for count in counts[1:]:
                np.testing.assert_array_equal(count, counts[0])
    return rows


def prepare():
    old = parent.frozen(PARENT)
    if read(PARENT / 'status.json')['phase'] != 'complete':
        raise ValueError('Require completed pilot014.')
    study = OUTPUT / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (study / 'scores').mkdir(parents=True, exist_ok=False)
    config = {'experiment_id': ID, 'parent_study': str(PARENT), 'seeds': old['seeds'],
              'selector': old['selector'], 'primary_fraction': .25, 'fractions': list(FRACTIONS),
              'confidence_bin_edges': BINS.tolist(), 'threads': 2,
              'scope': 'Retrospective diagnostic of reused hidden training inputs; no independent test or learning-utility claim.'}
    write_json(study / 'config.json', config)
    sources = {name: sha256(PROJECT / name) for name in SOURCES}
    for name in SOURCES:
        path = study / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(PROJECT / name, path)
    freeze = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'config_sha256': sha256(study / 'config.json'), 'source_sha256': sources,
              'parent_sha256': {str(p): sha256(p) for p in PARENT.rglob('*')
                                if p.is_file() and 'plot_cache' not in p.parts}}
    write_json(study / 'freeze.json', freeze)
    (study / 'freeze.sha256').write_text(sha256(study / 'freeze.json') + '\n')
    write_json(study / 'status.json', {'phase': 'prepared'})
    write_json(OUTPUT / 'latest.json', {'study': str(study)})
    return study


def frozen(study):
    study = Path(study)
    if sha256(study / 'freeze.json') != (study / 'freeze.sha256').read_text().strip():
        raise ValueError('Diagnostic freeze changed.')
    f = read(study / 'freeze.json')
    if sha256(study / 'config.json') != f['config_sha256']:
        raise ValueError('Diagnostic settings changed.')
    for name, digest in f['source_sha256'].items():
        if sha256(study / 'source' / name) != digest or (name.endswith('.py') and sha256(PROJECT / name) != digest):
            raise ValueError('Frozen diagnostic source changed.')
    if any(sha256(Path(name)) != digest for name, digest in f['parent_sha256'].items()):
        raise ValueError('Parent artifact changed.')
    parent.frozen(PARENT)
    return read(study / 'config.json')


def score(study):
    study = Path(study); config = frozen(study)
    if read(study / 'status.json')['phase'] != 'prepared':
        raise ValueError('Preserve existing diagnostic scores; do not restart.')
    if any((study / 'scores').iterdir()):
        raise ValueError('Preserve partial scores; use a separate declared run after a failure.')
    torch.set_num_threads(config['threads'])
    started = time.perf_counter()
    for seed in config['seeds']:
        with np.load(PARENT / f'targets_seed_{seed}.npz', allow_pickle=False) as t:
            scores, info = candidate_scores(t, config['selector'])
        model, _ = parent.load_model(PARENT / f'supervised160_seed_{seed}/model.pt')
        pool = ep.LearnerPool.load(PARENT / f'learner/seed_{seed}.npz')
        np.testing.assert_array_equal(scores['u_rows'], pool.unlabeled)
        scores['student_p'] = ep.probabilities(ep.infer(model, pool.x[pool.unlabeled])[0][0])
        np.savez_compressed(study / 'scores' / f'seed_{seed}.npz', **scores)
        write_json(study / 'scores' / f'selections_{seed}.json', selections(scores, seed))
        write_json(study / 'scores' / f'anchors_{seed}.json', info)
    freeze = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'files': {str(p.relative_to(study)): sha256(p) for p in (study / 'scores').iterdir()},
              'score_seconds': time.perf_counter() - started,
              'oracle_not_opened_in_scoring': True, 'endpoint_feature_or_score_payloads_loaded': 0,
              'new_fits': 0}
    write_json(study / 'score_freeze.json', freeze)
    write_json(study / 'status.json', {'phase': 'scores_complete'})
    return freeze


def analysis_ready(study):
    config = frozen(study)
    if read(study / 'status.json')['phase'] != 'scores_complete' or (study / 'oracle_access.json').exists():
        raise ValueError('Freeze all scores before oracle analysis; preserve completed access.')
    f = read(study / 'score_freeze.json')
    if any(sha256(study / name) != digest for name, digest in f['files'].items()):
        raise ValueError('Scores or selected rows changed before oracle access.')
    return config


def measure(chosen, pred, truth):
    n = len(chosen)
    errors = int(np.sum(pred[chosen] != truth[chosen]))
    return {'retained': n, 'errors': errors, 'error_rate': errors/n if n else None}


def analyze(study):
    study = Path(study); config = analysis_ready(study)
    write_json(study / 'oracle_access.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
              'role': 'Retrospective diagnostics after all rankings froze; no learning or selection feedback.'})
    out = study / 'analysis'; out.mkdir(exist_ok=False)
    with np.load(PARENT / 'oracle.npz', allow_pickle=False) as a:
        oracle = a['y'].copy()
    registry = read(PARENT / 'registry.json')['records']
    stage_rows, pred_rows, candidate_rows, ranking_rows, class_rows, proxy_rows = [], [], [], [], [], []
    ids = defaultdict(set)
    for seed in config['seeds']:
        with np.load(study / 'scores' / f'seed_{seed}.npz', allow_pickle=False) as a:
            d = {k: a[k].copy() for k in a.files}
        y = oracle[d['u_rows']]
        correct = d['pred'] == y
        student_correct = d['student_p'].argmax(1) == y
        stage = first_failure(d['agree'], d['base'], d['profile_supported'], d['profile_inside'])
        np.testing.assert_array_equal(stage == 4, d['profile_accepted'])
        for c, label in enumerate(parent.LABELS):
            m = y == c
            confident_correct = d['confidence'][m & correct & d['agree']]
            row = {'seed': seed, 'actual': label, 'total': int(m.sum()),
                   'teacher_a_correct': int(np.sum(m & (d['teacher_a'] == y))),
                   'teacher_b_correct': int(np.sum(m & (d['teacher_b'] == y))),
                   'ensemble_correct': int(np.sum(m & correct)),
                   'agreement': int(np.sum(m & d['agree'])),
                   'correct_agreement': int(np.sum(m & d['agree'] & correct)),
                   'wrong_agreement': int(np.sum(m & d['agree'] & ~correct)),
                   'confidence_accepted': int(np.sum(m & d['base'])),
                   'confidence_correct': int(np.sum(m & d['base'] & correct)),
                   'profile_correct': int(np.sum(m & d['profile_accepted'] & correct)),
                   'shared_support_agreement': int(np.sum(m & d['agree'] & d['shared_support']))}
            row.update({name: int(np.sum(m & (stage == i))) for i, name in enumerate(STAGES)})
            row.update({f'correct_agreement_confidence_q{q}': float(np.quantile(confident_correct, q/100)) if len(confident_correct) else None for q in (0,25,50,75,100)})
            assert sum(row[k] for k in STAGES) == row['total']
            stage_rows.append(row)
            p = d['pred'] == c
            pred_rows.append({'seed': seed, 'predicted': label, 'total': int(p.sum()),
                              'agreement': int(np.sum(p & d['agree'])), 'confidence_accepted': int(np.sum(p & d['base'])),
                              'profile_accepted': int(np.sum(p & d['profile_accepted'])),
                              'profile_supported_agreement': int(np.sum(p & d['agree'] & d['profile_supported'])),
                              'shared_supported_agreement': int(np.sum(p & d['agree'] & d['shared_support']))})
        for i, source_index in enumerate(d['u_rows']):
            raw = registry[int(source_index)]
            candidate_rows.append({'seed': seed, 'file': raw['file'], 'session': raw['session'],
                'actual': parent.LABELS[int(y[i])], 'proposed': parent.LABELS[int(d['pred'][i])],
                'teacher_a': parent.LABELS[int(d['teacher_a'][i])], 'teacher_b': parent.LABELS[int(d['teacher_b'][i])],
                'confidence': float(d['confidence'][i]), 'correct': bool(correct[i]),
                'first_failure': STAGES[int(stage[i])], 'profile_distance': float(d['profile_distance'][i]) if np.isfinite(d['profile_distance'][i]) else None,
                'embedding_distance': float(d['embedding_distance'][i]) if np.isfinite(d['embedding_distance'][i]) else None,
                'shared_support': bool(d['shared_support'][i]),
                'student_correct': bool(student_correct[i]), 'student_confidence': float(d['student_p'][i].max())})
        for selection in read(study / 'scores' / f'selections_{seed}.json'):
            chosen = np.array(selection['rows'], dtype=np.int64)
            key = (selection['domain'], selection['fraction'], selection['method'])
            ids[key].update(registry[int(d['u_rows'][i])]['file'] for i in chosen)
            ranking_rows.append({'seed': seed, **{k:v for k,v in selection.items() if k!='rows'}, **measure(chosen,d['pred'],y)})
            for c, label in enumerate(parent.LABELS):
                group = chosen[d['pred'][chosen] == c]
                class_rows.append({'seed': seed, 'domain': key[0], 'fraction': key[1], 'method': key[2],
                                   'predicted': label, **measure(group,d['pred'],y)})
        proxy_groups = {'profile_accepted': d['profile_accepted'], 'embedding_accepted': d['embedding_accepted'],
                        'confidence_accepted': d['base'],
                        'correct_agreement_not_profile_accepted': correct & d['agree'] & ~d['profile_accepted']}
        for name, mask in proxy_groups.items():
            n = int(mask.sum())
            proxy_rows.append({'seed': seed, 'group': name, 'count': n,
                               'student_correct': int(np.sum(mask & student_correct)),
                               'student_high_confidence': int(np.sum(mask & (d['student_p'].max(1) >= .95))),
                               'mean_student_confidence': float(d['student_p'][mask].max(1).mean()) if n else None})
    for name, rows in (('true_class_stages',stage_rows), ('predicted_class_stages',pred_rows),
                       ('candidates',candidate_rows),('ranking_results',ranking_rows),
                       ('ranking_classes',class_rows),('student_proxies',proxy_rows)):
        write_csv(out / (name+'.csv'), rows)
    pooled = []
    for key, unique in ids.items():
        members = [r for r in ranking_rows if (r['domain'],r['fraction'],r['method'])==key]
        n, errors = sum(r['retained'] for r in members), sum(r['errors'] for r in members)
        available = sum(r['available'] for r in members)
        pooled.append({'domain':key[0], 'fraction':key[1], 'method':key[2],
                       'retained':n,'errors':errors,'error_rate':errors/n if n else None,
                       'available':available,'actual_coverage':n/available if available else None,
                       'unique_clips':len(unique)})
    paired = []
    for domain in ('class_quota','confidence_matched','frozen_gate'):
        for method in ('profile','embedding','random'):
            members = [r for r in ranking_rows if r['domain']==domain and r['fraction']==.25]
            diffs = []
            for seed in config['seeds']:
                a = next(r for r in members if r['seed']==seed and r['method']==method)
                b = next(r for r in members if r['seed']==seed and r['method']=='confidence')
                assert a['retained']==b['retained']
                diffs.append(a['errors']-b['errors'])
            paired.append({'domain':domain,'method':method,'error_count_minus_confidence_by_seed':diffs,
                           'pooled_error_count_difference':sum(diffs)})
    summary = {'scope':config['scope'], 'primary_fraction':.25, 'seed_clip_pairs':len(candidate_rows),
               'unique_hidden_clips':len({r['file'] for r in candidate_rows}),
               'rankings':pooled,'paired_at_25_percent':paired,'new_fits':0,'endpoint_payloads_loaded':0}
    write_csv(out / 'ranking_pooled.csv',pooled); write_json(out / 'summary.json',summary)
    make_report(study, summary, stage_rows, proxy_rows)
    plot(study, summary, stage_rows, proxy_rows)
    write_json(study / 'status.json', {'phase':'complete'})
    return summary


def make_report(study, summary, stages, proxies):
    lines = ['# Diagnostic 015: coverage and ranking information','',summary['scope'],'',
        'Five saved 160-label seeds; hidden training inputs only. Scores and selected IDs froze before this run\'s oracle access; outcomes were already partly known from earlier pilots. No fitting, new labels, threshold tuning or evaluation-endpoint rescoring.','',
        '| Actual command | Hidden pairs | Ensemble correct | Correct agreement | Pass confidence | Accepted profile | Disagree / low confidence / anchors / radius |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for label in parent.LABELS:
        rows=[r for r in stages if r['actual']==label]
        total=lambda k:sum(r[k] for r in rows)
        lines.append(f"| {label} | {total('total')} | {total('ensemble_correct')} | {total('correct_agreement')} | {total('confidence_accepted')} | {total('accepted')} | {' / '.join(str(total(k)) for k in STAGES[:-1])} |")
    lines += ['', 'First failures are disjoint. Pass-confidence and accepted columns may include wrong predictions; correct/incorrect agreement and per-seed confidence quantiles are in the CSVs.','',
        '| Domain at fixed 25% cell quota | Ranker | Errors / retained pairs | Unique clips | Available pairs |',
        '| --- | --- | ---: | ---: | ---: |']
    for r in summary['rankings']:
        if r['fraction']==.25:
            lines.append(f"| {r['domain']} | {r['method']} | {r['errors']} / {r['retained']} | {r['unique_clips']} | {r['available']} |")
    lines += ['', 'Quotas are ceil(fraction × cell size), so actual retained coverage can exceed 25%. Every ranker shares the same candidates and predicted-class counts; confidence_matched also shares fixed confidence-bin counts. It does not exactly match continuous confidence. Unsupported candidates are excluded from every ranker and reported in the stage tables.','',
        '| Original selection group | Pairs | Already correct in supervised160 | Student confidence >=.95 |','| --- | ---: | ---: | ---: |']
    for group in sorted({r['group'] for r in proxies}):
        rows=[r for r in proxies if r['group']==group]
        lines.append(f"| {group} | {sum(r['count'] for r in rows)} | {sum(r['student_correct'] for r in rows)} | {sum(r['student_high_confidence'] for r in rows)} |")
    lines += ['', 'Already-correct examples can still regularize or improve a model; this is only a redundancy proxy. Recorded-label ranking accuracy is not training utility. Repeated seed–clip pairs and overlapping seeds are not independent participants or trials. No independent test, robot action, broad reliability, causal learning gain, novelty or label/compute saving is established.','',
        'All fixed 10/25/50/100% coverage results, confidence-bin controls, per-class/per-seed tables, selected IDs, raw candidate scores, parent/source hashes and verification are retained. No favorable coverage point or random seed selected.']
    (study / 'REPORT.md').write_text('\n'.join(lines)+'\n')


def plot(study, summary, stages, proxies):
    os.environ.setdefault('MPLCONFIGDIR',str(study / 'analysis/plot_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(15,10),constrained_layout=True)
    colors={'confidence':'#707a86','profile':'#167d8d','embedding':'#8063a5','random':'#ba765c'}
    a=axes[0,0]; bottom=np.zeros(8)
    for name,color in zip(STAGES,('#c7d4dc','#e9c581','#b9a1bb','#87b2ba','#268878')):
        values=np.array([sum(r[name] for r in stages if r['actual']==label)/sum(r['total'] for r in stages if r['actual']==label)*100 for label in parent.LABELS])
        a.bar(range(8),values,bottom=bottom,label=name.replace('_',' '),color=color); bottom+=values
    a.set(xticks=range(8),xticklabels=['F','B','L','R','STOP','NONE','UP','DOWN'],ylim=(0,100),ylabel='Hidden true-class pairs (%)',title='A  Where the original selector loses coverage')
    a.legend(fontsize=8,ncol=2,loc='lower left')
    a=axes[0,1]
    for method in METHODS:
        rows=sorted((r for r in summary['rankings'] if r['domain']=='class_quota' and r['method']==method),key=lambda r:r['fraction'])
        a.plot([r['actual_coverage']*100 for r in rows],[r['error_rate']*100 if r['error_rate'] is not None else np.nan for r in rows],'-o',label=method,color=colors[method])
    a.set(xlabel='Retained supported/agreed pool (%)',ylabel='Recorded-label error (%)',title='B  Same predicted-class quotas; all frozen fractions')
    a.legend(fontsize=9)
    a=axes[1,0]
    rows=[next(r for r in summary['rankings'] if r['domain']=='confidence_matched' and r['fraction']==.25 and r['method']==m) for m in METHODS]
    values=[100*r['error_rate'] if r['error_rate'] is not None else 0 for r in rows]
    a.bar(range(4),values,color=[colors[m] for m in METHODS]); top=max(5,max(values)*1.3)
    a.set(xticks=range(4),xticklabels=METHODS,ylim=(0,top),ylabel='Recorded-label error (%)',title='C  Also match fixed confidence-bin quotas')
    for i,r in enumerate(rows): a.text(i,values[i]+top*.03,f"{r['errors']}/{r['retained']}",ha='center')
    a=axes[1,1]; groups=('profile_accepted','embedding_accepted','confidence_accepted','correct_agreement_not_profile_accepted')
    counts=[sum(r['count'] for r in proxies if r['group']==g) for g in groups]
    correct=[sum(r['student_correct'] for r in proxies if r['group']==g) for g in groups]
    values=[100*c/n if n else 0 for c,n in zip(correct,counts)]
    a.bar(range(4),values,color=['#167d8d','#8063a5','#707a86','#9bb6c3'])
    a.set(xticks=range(4),xticklabels=['Profile\naccepted','Embedding\naccepted','Confidence\naccepted','Correct agreed\nnot profile-selected'],ylim=(0,115),ylabel='Already correct in supervised160 (%)',title='D  A redundancy proxy, not causal usefulness')
    for i,(c,n) in enumerate(zip(correct,counts)): a.text(i,values[i]+2,f'{c}/{n}',ha='center')
    fig.suptitle('Diagnostic 015 · saved 160-label models · hidden training inputs only',fontsize=15)
    fig.supxlabel('Retrospective · repeated seed–clip pairs · zero new fits · no evaluation-endpoint rescoring',fontsize=10)
    fig.savefig(study/'analysis/figure.png',dpi=160); fig.savefig(study/'analysis/figure.pdf'); plt.close(fig)


def verify(study):
    study=Path(study); config=frozen(study)
    if read(study/'status.json')['phase']!='complete':
        raise ValueError('Finish the diagnostic before verification.')
    torch.set_num_threads(config['threads']); vectors=0
    for seed in config['seeds']:
        with np.load(PARENT/f'targets_seed_{seed}.npz',allow_pickle=False) as a:
            expected,info=candidate_scores(a,config['selector'])
        with np.load(study/'scores'/f'seed_{seed}.npz',allow_pickle=False) as a:
            saved={k:a[k].copy() for k in a.files}
        for k,v in expected.items(): np.testing.assert_allclose(v,saved[k],rtol=1e-6,atol=1e-7)
        assert info==read(study/'scores'/f'anchors_{seed}.json')
        assert selections(expected,seed)==read(study/'scores'/f'selections_{seed}.json')
        pool=ep.LearnerPool.load(PARENT/f'learner/seed_{seed}.npz')
        model,_=parent.load_model(PARENT/f'supervised160_seed_{seed}/model.pt')
        p=ep.probabilities(ep.infer(model,pool.x[pool.unlabeled])[0][0])
        np.testing.assert_allclose(p,saved['student_p'],rtol=1e-6,atol=1e-7); vectors+=len(p)
    f=read(study/'score_freeze.json')
    assert all(sha256(study/name)==digest for name,digest in f['files'].items())
    assert f['created_utc']<=read(study/'oracle_access.json')['created_utc']
    result={'parent_artifacts_intact':True,'seed_score_bundles_reproduced':5,
            'ranking_selections_reproduced':180,'student_prediction_vectors_reproduced':vectors,
            'original_profile_and_embedding_weights_reproduced':True,'scores_frozen_before_oracle_access':True,
            'new_fits':0,'endpoint_feature_or_score_payloads_loaded':0}
    write_json(study/'verification.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','score','analyze','verify'))
    parser.add_argument('study',type=Path,nargs='?')
    args=parser.parse_args()
    if args.action=='prepare': print(prepare())
    else:
        if args.study is None: parser.error('Provide the frozen diagnostic path.')
        result={'score':score,'analyze':analyze,'verify':verify}[args.action](args.study)
        print(json.dumps(result if args.action!='analyze' else {'phase':'complete','seed_clip_pairs':result['seed_clip_pairs']},indent=2))


if __name__=='__main__':
    main()
