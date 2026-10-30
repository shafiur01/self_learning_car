"""Separate post-freeze evaluation and oracle diagnostics for018."""

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import online_softmatch as study
from .run import PROJECT,sha256,write_json


def csv_out(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def fit_guard(folder):
    freeze=study.read(folder/'fit_freeze.json')
    assert freeze['fits']==15
    for name,h in freeze['files'].items():
        assert sha256(folder/name)==h,name
    return freeze


def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['experiment_id']==study.ID and saved['features']==149
    assert saved['parameters']==study.ep.PARAMETERS
    assert saved['labels']==list(study.parent.LABELS)
    model=study.ep.Model();model.load_state_dict(saved['state_dict']);model.eval()
    return model,saved


def verify(folder):
    config=study.frozen(folder);fit_guard(folder)
    torch.set_num_threads(2)
    trials=study.read(folder/'trials.json')
    assert len(trials)==15
    with np.load(study.PARENT/'selection.npz',allow_pickle=False) as z:
        vx,vy=z['x'].copy(),z['y'].copy()
    vectors=0;traces=0;parent_matches=0
    for seed in config['seeds']:
        group=[r for r in trials if r['seed']==seed]
        for key in ('initial_hash','warmup_hash','labeled_draws','labeled_dropout'):
            assert len({r[key] for r in group})==1,(seed,key)
        ssl=[r for r in group if r['method']!='supervised160']
        assert len({r['hidden_draws'] for r in ssl})==1
        for r in group:
            path=folder/f'{r["method"]}_seed_{seed}'
            access=study.read(path/'access.json')
            assert access['loaded']==access['allowed']
            assert r['training_class_counts']==[20]*8
            assert r['optimizer_steps']==560 and r['validation_checks']==28
            history=study.read(path/'history.json')
            assert r['selected_step']==min(history,key=lambda x:x['selection_loss'])['step']
            with np.load(path/'development.npz',allow_pickle=False) as z:
                for kind in ('selected','final'):
                    model,saved=load_model(path/f'{kind}.pt')
                    p=study.predict(model,vx)
                    np.testing.assert_allclose(p,z[kind],rtol=1e-6,atol=1e-7)
                    metrics=study.METRICS(p,vy)
                    assert metrics==r['development'][kind]
                    assert saved['step']==(560 if kind=='final' else r['selected_step'])
                    vectors+=len(p)
                    if kind=='selected' and r['method']=='supervised160':
                        old=study.read(study.PARENT/f'supervised160_seed_{seed}/metrics.json')
                        assert study.ep.state_hash(model)==old['state_sha256']
                        assert r['labeled_draws']==old['batch_indices_sha256']
                        assert r['initial_hash']==old['initial_state_sha256']
                        parent_matches+=1
            if r['method']=='supervised160':continue
            state=study.SoftWeights(config['momentum'])
            with np.load(path/'trace.npz',allow_pickle=False) as z:
                assert z['p'].shape==(420,32,8)
                assert z['rows'].shape==z['w'].shape==z['target'].shape==(420,32)
                for i,p in enumerate(z['p']):
                    targets=torch.from_numpy(p).argmax(1)
                    weights=(torch.from_numpy(p).max(1).values>=.95).float()
                    if r['method']=='online_softmatch':targets,weights=state.update(torch.from_numpy(p))
                    np.testing.assert_array_equal(targets.numpy(),z['target'][i])
                    np.testing.assert_allclose(weights.numpy(),z['w'][i],rtol=1e-6,atol=1e-7)
                    traces+=32
                with np.load(study.PARENT/f'learner/seed_{seed}.npz',allow_pickle=False) as pool:
                    assert np.isin(z['rows'],pool['unlabeled']).all()
                    rng=torch.Generator().manual_seed(seed+300+3_000_000)
                    draws=np.stack([torch.randint(245,(32,),generator=rng).numpy() for _ in range(420)])
                    np.testing.assert_array_equal(z['rows'],pool['unlabeled'][draws])
    result={'models':30,'development_probability_vectors_reproduced':vectors,
            'online_trace_targets_weights_reproduced':traces,'supervised_parent_states_match':parent_matches,
            'paired_initial_warmup_labeled_streams':True,'training_payload_allowlists_verified':True,
            'parent_artifacts_intact':True,'no_endpoint_payload':True}
    write_json(folder/'verification.json',result)
    return result


def analyze(folder):
    config=study.frozen(folder);frozen=fit_guard(folder)
    assert study.read(folder/'status.json')=={'phase':'fits_complete','completed_fits':15}
    assert not (folder/'oracle_access.json').exists()
    verification=verify(folder)
    records=study.read(study.PARENT/'registry.json')
    with np.load(study.PARENT/'selection.npz',allow_pickle=False) as z:vy=z['y'].copy()
    # All models and all training pseudo-labels already frozen; no tuning follows.
    write_json(folder/'oracle_access.json',{'created_utc':study.utc(),
               'role':'Retrospective hidden-training diagnostic after all15fits; no feedback to learning.'})
    assert frozen['created_utc']<study.read(folder/'oracle_access.json')['created_utc']
    with np.load(study.PARENT/'oracle.npz',allow_pickle=False) as z:oracle=z['y'].copy()
    trials=study.read(folder/'trials.json')
    per_seed=[];failures=[];pseudo=[]
    for r in trials:
        path=folder/f'{r["method"]}_seed_{r["seed"]}'
        with np.load(path/'development.npz',allow_pickle=False) as z:
            for kind in ('selected','final'):
                metrics=r['development'][kind]
                per_seed.append({'seed':r['seed'],'method':r['method'],'checkpoint':kind,
                    'step':r['selected_step'] if kind=='selected' else 560,
                    **{k:metrics[k] for k in ('accuracy','macro_f1','nll','no_command_to_motion_rate','stop_recall')},
                    'fit_seconds':r['fit_seconds'],'worker_peak_rss_bytes':r['peak_rss_bytes']})
                p=z[kind];pred=p.argmax(1)
                for i in np.flatnonzero(pred!=vy):
                    failures.append({'seed':r['seed'],'method':r['method'],'checkpoint':kind,
                                     'file':records['selection_files'][i],
                                     'actual':study.parent.LABELS[vy[i]],'predicted':study.parent.LABELS[pred[i]],
                                     'confidence':float(p[i].max())})
        if r['method']=='supervised160':continue
        with np.load(path/'trace.npz',allow_pickle=False) as z:
            actual=oracle[z['rows']];target=z['target'];weight=z['w'].astype(float)
            log=study.read(path/'training_log.json')[140:]
            coefficient=np.array([row['coefficient'] for row in log])[:,None]
            for phase,mask in (('all',np.ones(420,dtype=bool)),('ramp',np.arange(420)<140),('full',np.arange(420)>=140)):
                for axis,labels in (('actual',actual),('predicted',target)):
                    for c,name in enumerate(study.parent.LABELS):
                        keep=(labels==c)&mask[:,None]
                        correct=actual==target
                        motion=np.isin(target,[0,1,2,3,6,7])
                        w=np.where(keep,weight,0)
                        pseudo.append({'seed':r['seed'],'method':r['method'],'phase':phase,'axis':axis,'class':name,
                            'exposures':int(keep.sum()),'mass':float(w.sum()),
                            'wrong_mass':float(w[~correct].sum()),
                            'movement_from_none_mass':float(w[(actual==5)&motion].sum()),
                            'coefficient_adjusted_mass':float((w*coefficient).sum()),
                            'coefficient_adjusted_wrong_mass':float((w*coefficient)[~correct].sum())})
    summary=[]
    for kind in ('selected','final'):
        for method in config['methods']:
            rows=[r for r in per_seed if r['checkpoint']==kind and r['method']==method]
            out={'method':method,'checkpoint':kind}
            for metric in ('accuracy','macro_f1','nll','no_command_to_motion_rate','stop_recall'):
                out[metric]=float(np.mean([r[metric] for r in rows]))
                out[metric+'_sd']=float(np.std([r[metric] for r in rows],ddof=1))
            summary.append(out)
    chosen={r['method']:r for r in summary if r['checkpoint']=='selected'}
    a,b=chosen['online_softmatch'],chosen['online_confidence']
    differences=[]
    for seed in config['seeds']:
        rows={r['method']:r for r in per_seed if r['checkpoint']=='selected' and r['seed']==seed}
        differences.append(rows['online_softmatch']['macro_f1']-rows['online_confidence']['macro_f1'])
    gate={'mean_macro_f1_difference':a['macro_f1']-b['macro_f1'],
          'paired_seed_differences':differences,'positive_seeds':sum(d>0 for d in differences),
          'no_command_to_motion_difference':a['no_command_to_motion_rate']-b['no_command_to_motion_rate'],
          'stop_recall_difference':a['stop_recall']-b['stop_recall'],
          'mean_f1_improves':a['macro_f1']>b['macro_f1'],
          'at_least_three_positive':sum(d>0 for d in differences)>=3,
          'background_not_worse':a['no_command_to_motion_rate']<=b['no_command_to_motion_rate'],
          'stop_not_worse':a['stop_recall']>=b['stop_recall']}
    gate['passed']=all(gate[k] for k in ('mean_f1_improves','at_least_three_positive','background_not_worse','stop_not_worse'))
    gate['f1_difference_vs_supervised']=a['macro_f1']-chosen['supervised160']['macro_f1']
    csv_out(folder/'development_per_seed.csv',per_seed)
    csv_out(folder/'pseudo_label_diagnostics.csv',pseudo)
    if failures:csv_out(folder/'development_failures.csv',failures)
    write_json(folder/'summary.json',{'metrics':summary,'continuation_gate':gate,
               'fits':15,'fit_seconds':sum(r['fit_seconds'] for r in trials),
               'subprocess_wall_seconds':frozen['elapsed_subprocess_seconds'],
               'maximum_worker_peak_rss_bytes':max(r['peak_rss_bytes'] for r in trials),
               'selected_before_ssl':{m:sum(r['selected_before_ssl'] for r in trials if r['method']==m) for m in config['methods']}})
    lines=['# 018 — Online pseudo-label control experiment','',
           '160 revealed +245 hidden training inputs per seed;48 inspected development clips. No96 rescoring.',
           'This is a paper-based SoftMatch adaptation with classifier dropout, supervised warm-up, changed statistics timescale and no parameter EMA; not a full paper reproduction.','',
           '| Checkpoint | Method | Accuracy | Macro F1 | NONE→movement | STOP recall |',
           '|---|---|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f'| {row["checkpoint"]} | {row["method"]} | {row["accuracy"]:.2%} | {row["macro_f1"]:.4f} | {row["no_command_to_motion_rate"]:.2%} | {row["stop_recall"]:.2%} |')
    lines+=['',f'Continuation gate: **{gate["passed"]}**; paired macroF1 differences {differences}.',
            '',f'Fitting-loop seconds: {sum(r["fit_seconds"] for r in trials):.2f}; subprocess wall {frozen["elapsed_subprocess_seconds"]:.2f}. This includes development checks; review/preparation/audit costs are additional.',
            '', 'All per-seed/step, failure and training pseudo-label records are in the adjacent CSV/JSON files. Exposure counts repeat recordings. No independence, actual label savings, robotics safety or novelty established.']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(folder/'status.json',{'phase':'complete','completed_fits':15})
    return {'summary':summary,'gate':gate,'verification':verification}
