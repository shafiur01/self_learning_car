"""Reproduce experiment016 and report every fixed factor/seed and limitation."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import input_conditioning as run
from .run import sha256,write_json


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def verify(study):
    study=Path(study);config=run.frozen(study)
    if run.read(study/'status.json')!={'phase':'complete','completed_fits':60}:
        raise ValueError('Require a complete60-fit evaluation.')
    torch.set_num_threads(config['threads'])
    trials=run.read(study/'trials.json');fitfreeze=run.read(study/'fit_freeze.json')
    assert len(trials)==60 and sha256(study/'trials.json')==fitfreeze['trials_sha256']
    with np.load(run.PARENT/'selection.npz',allow_pickle=False) as a:vx,vy=a['x'],a['y']
    with np.load(run.PARENT/'endpoint/features.npz',allow_pickle=False) as a:ex,ey=a['x'],a['y']
    with np.load(run.PARENT/'oracle.npz',allow_pickle=False) as a:oracle=a['y']
    results=run.read(study/'evaluation/results.json')
    assert len(results)==20
    vectors=0;streams={};initial={};raw_matches=0
    for seed in config['seeds']:
        pool=run.ep.LearnerPool.load(run.PARENT/f'learner/seed_{seed}.npz')
        for role in run.ROLES:
            normalizer=run.read(study/'normalizers'/f'{role}_{seed}.json')
            assert normalizer==run.role_scales(pool,role)
        for condition in run.CONDITIONS:
            expected={};gate={}
            for role in run.ROLES:
                trial=next(t for t in trials if t['seed']==seed and t['role']==role and t['condition']==condition)
                folder=study/trial['folder'];model,c=run.load_model(folder/'model.pt')
                stats_path=study/'normalizers'/f'{role}_{seed}.json';stats=run.read(stats_path)
                assert sha256(folder/'model.pt')==trial['checkpoint_sha256']==fitfreeze['checkpoints'][trial['folder']]
                assert run.ep.state_hash(model)==trial['state_sha256']
                for name,value in {'condition':condition,'role':role,'seed':seed,'normalizer_sha256':sha256(stats_path),
                                   'best_step':trial['best_step'],'config_sha256':sha256(study/'config.json'),
                                   'freeze_sha256':sha256(study/'freeze.json')}.items():assert c[name]==value
                assert run.read(folder/'metrics.json')==trial
                hist=run.read(folder/'history.json');best=min(hist,key=lambda h:h['selection_loss'])
                assert len(hist)==28 and best['step']==trial['best_step']
                assert [h['step'] for h in hist]==list(range(20,561,20))
                p=run.ep.probabilities(run.ep.infer(model,run.transform(vx,stats,condition))[0][0])
                np.testing.assert_allclose(p,run.read(folder/'validation_probabilities.json'),rtol=1e-6,atol=1e-7)
                metric=run.parent.old.motion.previous.fitting.metric_set
                assert metric(p,vy)==trial['validation']
                assert np.isclose(best['selection_loss'],trial['validation']['nll'],rtol=1e-6,atol=1e-7)
                key=(seed,role);streams.setdefault(key,set()).add((trial['batch_indices_sha256'],trial['mask_sha256']))
                initial.setdefault(key,set()).add(trial['initial_state_sha256'])
                assert trial['training_class_counts']==([20]*8 if role=='student' else [10]*8)
                assert not trial['active_unsupervised_loss'];vectors+=48
                if role=='student':
                    expected['student_endpoint']=run.ep.probabilities(run.ep.infer(model,run.transform(ex,stats,condition))[0][0])
                else:
                    anchors=pool.teacher_b if role=='teacher_a' else pool.teacher_a
                    ax=run.transform(pool.x[anchors],stats,condition)
                    expected[role+'_anchor']=run.ep.probabilities(run.ep.infer(model,ax)[0][0])
                    expected[role+'_hidden']=run.ep.probabilities(run.ep.infer(model,run.transform(pool.x[pool.unlabeled],stats,condition))[0][0])
                    gate[role]=run.gate_stats(model,ax)
                if condition=='raw_clock':
                    prev=run.read(run.PARENT/'trials.json')
                    method='supervised160' if role=='student' else 'teacher_subset'
                    old=next(t for t in prev if t['seed']==seed and t['method']==method and (role=='student' or t['side']==role[-1]))
                    assert trial['state_sha256']==old['state_sha256'];raw_matches+=1
            with np.load(study/'evaluation'/f'{condition}_seed_{seed}.npz',allow_pickle=False) as a:
                assert set(a.files)==set(expected)
                for name,value in expected.items():
                    np.testing.assert_allclose(value,a[name],rtol=1e-6,atol=1e-7);vectors+=len(value)
            result=next(r for r in results if r['seed']==seed and r['condition']==condition)
            assert result['gates']==gate
            q=(expected['teacher_a_hidden']+expected['teacher_b_hidden'])/2
            assert metric(q,oracle[pool.unlabeled])==result['teacher_hidden_mean']
            assert metric(expected['student_endpoint'],ey)==result['student_endpoint']
            ap=np.r_[expected['teacher_a_anchor'],expected['teacher_b_anchor']]
            ay=np.r_[pool.labels_for(pool.teacher_b),pool.labels_for(pool.teacher_a)]
            assert metric(ap,ay)==result['teacher_cross_session_anchors']
    assert all(len(v)==1 for v in streams.values()) and all(len(v)==1 for v in initial.values())
    answer={'checkpoints_verified':60,'prediction_vectors_reproduced':vectors,'own_label_only_scale_fits_verified':15,
            'raw_parent_state_matches':raw_matches,'all_role_initial_batch_mask_streams_paired':True,
            'parent_artifacts_intact':True,'new_data_or_activation':False}
    write_json(study/'verification.json',answer);return answer


def report(study):
    study=Path(study);config=run.frozen(study)
    if run.read(study/'status.json')['phase']!='complete':raise ValueError('Finish all conditions before reporting.')
    out=study/'analysis';out.mkdir(exist_ok=False)
    results=run.read(study/'evaluation/results.json');trials=run.read(study/'trials.json')
    fields=('accuracy','macro_f1','nll','no_command_to_motion_rate','stop_recall')
    groups={};rows=[];classes=[];gates=[];costs=[]
    for condition in run.CONDITIONS:
        members=[r for r in results if r['condition']==condition]
        groups[condition]={}
        for endpoint in ('student_endpoint','teacher_cross_session_anchors','teacher_hidden_mean'):
            groups[condition][endpoint]={k:{'mean':float(np.mean([r[endpoint][k] for r in members])),
                'seed_sd':float(np.std([r[endpoint][k] for r in members],ddof=1))} for k in fields}
        for r in members:
            rows.append({'condition':condition,'seed':r['seed'],
                **{k:r['student_endpoint'][k] for k in fields},
                'anchor_macro_f1':r['teacher_cross_session_anchors']['macro_f1'],
                'teacher_hidden_macro_f1':r['teacher_hidden_mean']['macro_f1'],
                'sitting1_accuracy':r['sittings']['1']['accuracy'],'sitting2_accuracy':r['sittings']['2']['accuracy'],
                'admitted':r['admitted_count'],'admitted_errors':r['admitted_errors'],
                'admitted_up':r['admitted_predicted_counts']['UP'],
                'admitted_none':r['admitted_predicted_counts']['NO_COMMAND']})
            for endpoint in ('student_endpoint','teacher_cross_session_anchors','teacher_hidden_mean'):
                for label,values in r[endpoint]['per_class'].items():classes.append({'condition':condition,'seed':r['seed'],'endpoint':endpoint,'label':label,**values})
            for role,gs in r['gates'].items():
                for gate,values in gs.items():gates.append({'condition':condition,'seed':r['seed'],'role':role,'gate':gate,**values})
            fits=[t for t in trials if t['condition']==condition and t['seed']==r['seed']]
            costs.append({'condition':condition,'seed':r['seed'],
                          'student_fit_seconds':sum(t['fit_seconds'] for t in fits if t['role']=='student'),
                          'teacher_fit_seconds':sum(t['fit_seconds'] for t in fits if t['role']!='student')})
    by={(r['condition'],r['seed']):r for r in results};paired={}
    for name,a,b in (('primary_rms_clock_minus_raw_clock','rms_clock','raw_clock'),
                     ('rms_without_clock','rms_no_clock','raw_no_clock'),
                     ('clock_removal_raw','raw_no_clock','raw_clock'),
                     ('clock_removal_rms','rms_no_clock','rms_clock')):
        paired[name]={}
        for endpoint in ('student_endpoint','teacher_cross_session_anchors','teacher_hidden_mean'):
            d=[by[a,s][endpoint]['macro_f1']-by[b,s][endpoint]['macro_f1'] for s in config['seeds']]
            paired[name][endpoint]={'differences':d,'mean':float(np.mean(d)),'positive_pairs':sum(x>0 for x in d)}
        for sitting in ('1','2'):
            d=[by[a,s]['sittings'][sitting]['macro_f1']-by[b,s]['sittings'][sitting]['macro_f1'] for s in config['seeds']]
            paired[name]['sitting'+sitting]={'differences':d,'mean':float(np.mean(d))}
    interaction=[(by['rms_no_clock',s]['student_endpoint']['macro_f1']-by['raw_no_clock',s]['student_endpoint']['macro_f1'])-
                 (by['rms_clock',s]['student_endpoint']['macro_f1']-by['raw_clock',s]['student_endpoint']['macro_f1']) for s in config['seeds']]
    summary={'groups':groups,'paired_macro_f1':paired,'student_interaction':{'differences':interaction,'mean':float(np.mean(interaction))},
             'fit_seconds_total':float(sum(t['fit_seconds'] for t in trials)),'fresh_fits':60,'new_recordings':0,'scope':config['scope']}
    write_json(out/'summary.json',summary)
    for name,values in (('model_results',rows),('per_class_results',classes),('gate_saturation',gates),('costs',costs)):
        write_csv(out/(name+'.csv'),values)
    lines=['# Experiment016: feature conditioning and explicit-clock control','',config['scope'],'',
        '160 labels per student;80 per teacher. All eight commands represented. Each model\'s seven scale factors fit only its own labeled training inputs. Same48 development clips select weights; same96 historically inspected clips supply retrospective student comparison. Hidden245 labels used only after all60 fits freeze. No pseudo-label student training.','',
        '| Condition | Student96 accuracy (%) | MacroF1 | NONE→movement (%) | STOP recall (%) | Anchor teacher F1 | Hidden mean-teacher F1 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for c,g in groups.items():
        m=g['student_endpoint'];lines.append(f"| {c} | {m['accuracy']['mean']*100:.2f} | {m['macro_f1']['mean']:.4f} | {m['no_command_to_motion_rate']['mean']*100:.2f} | {m['stop_recall']['mean']*100:.2f} | {g['teacher_cross_session_anchors']['macro_f1']['mean']:.4f} | {g['teacher_hidden_mean']['macro_f1']['mean']:.4f} |")
    p=paired['primary_rms_clock_minus_raw_clock']['student_endpoint']
    lines+=['',f"Primary paired student macroF1 RMS−raw with clock: {p['mean']:+.6f}, positive in{p['positive_pairs']}/5 seeds. All paired contrasts/sittings and factorial interaction archived; no winning seed or condition activated.",'',
        '| Condition | Confidence-admitted pairs | Wrong labels | Predicted UP / NONE admissions |','| --- | ---: | ---: | ---: |']
    for c in run.CONDITIONS:
        m=[r for r in rows if r['condition']==c]
        lines.append(f"| {c} | {sum(r['admitted'] for r in m)} | {sum(r['admitted_errors'] for r in m)} | {sum(r['admitted_up'] for r in m)} / {sum(r['admitted_none'] for r in m)} |")
    lines+=['',f"60 numerical fit loops total{summary['fit_seconds_total']:.2f}s, excluding scaling/preparation/I/O/evaluation/report/prior work. Scale statistics add preprocessing state, not trainable LSTM parameters. No measured inference/memory/annotation saving.",'',
        'Fifteen raw control states must exactly reproduce parent014. Same within-role initial weights, batches, masks and optimizer updates across allfour conditions. No extra training, data, model size, gate or learning-rate tuning. Gates measured on known opposite-session anchors are descriptive; saturation changes do not prove a cause of classification changes.','',
        'Camera/hand earlier meant coordinate representation, not an isolated physical viewpoint experiment. Explicit clock removal leaves physical-time velocity features intact. This study cannot demonstrate time/viewpoint invariance, novel algorithm, independent generalization or robot-action reliability. STOP/NONE examples remain few and same-user.','',
        'Technical basis: [PyTorch LSTM equations](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html); [MediaPipe coordinate definitions](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python#handle_and_display_results). These motivate hypotheses, not a guarantee that normalization improves this task.']
    (study/'REPORT.md').write_text('\n'.join(lines)+'\n')
    plot(study,summary,rows,gates)
    return summary


def plot(study,summary,rows,gates):
    os.environ.setdefault('MPLCONFIGDIR',str(study/'analysis/plot_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    names=['Raw\nclock','RMS\nclock','Raw\nno clock','RMS\nno clock'];colors=['#9ab2c0','#167d8d','#c7a77d','#8b719f']
    a=axes[0,0];values=[summary['groups'][c]['student_endpoint']['accuracy']['mean']*100 for c in run.CONDITIONS]
    a.bar(range(4),values,color=colors);a.set(xticks=range(4),xticklabels=names,ylim=(0,110),ylabel='Accuracy (%)',title='A  Retrospective96 · same160 labels')
    for i,v in enumerate(values):a.text(i,v+2,f'{v:.2f}',ha='center')
    a=axes[0,1]
    contrasts=list(summary['paired_macro_f1'])
    for i,k in enumerate(contrasts):
        d=np.array(summary['paired_macro_f1'][k]['student_endpoint']['differences'])*100
        a.bar(i,d.mean(),color='#167d8d' if d.mean()>0 else '#bd858b');a.scatter(i+np.linspace(-.12,.12,5),d,color='#223747',zorder=3)
    a.axhline(0,color='gray',lw=1);a.set(xticks=range(4),xticklabels=['RMS gain\nclock','RMS gain\nno clock','Clock removal\nraw','Clock removal\nRMS'],ylabel='Paired macroF1 ×100',title='B  All fixed contrasts · shared clips/seeds')
    a=axes[1,0]
    for endpoint,label in (('teacher_cross_session_anchors','Opposite-session anchors'),('teacher_hidden_mean','Hidden training inputs')):
        a.plot(range(4),[summary['groups'][c][endpoint]['macro_f1']['mean']*100 for c in run.CONDITIONS],'-o',label=label)
    a.set(xticks=range(4),xticklabels=names,ylim=(0,105),ylabel='Teacher macroF1 ×100',title='C  Teacher recognition before pseudo-label training');a.legend(fontsize=9)
    a=axes[1,1]
    for gate in ('input','forget','candidate','output'):
        values=[np.mean([r['saturation_fraction'] for r in gates if r['condition']==c and r['gate']==gate])*100 for c in run.CONDITIONS]
        a.plot(range(4),values,'-o',label=gate)
    a.set(xticks=range(4),xticklabels=names,ylabel='Saturated activations (%)',title='D  Final teacher gates · descriptive, not causal');a.legend(fontsize=9)
    fig.suptitle('Experiment016 · feature units and explicit-clock controls',fontsize=15)
    fig.supxlabel('Same-user retrospective comparison · training-only scale estimates · no physical viewpoint or novelty claim',fontsize=10)
    fig.savefig(study/'analysis/figure.png',dpi=160);fig.savefig(study/'analysis/figure.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('verify','report'));p.add_argument('study',type=Path)
    args=p.parse_args();answer={'verify':verify,'report':report}[args.action](args.study)
    print(json.dumps(answer if args.action=='verify' else {'reported':True,'fresh_fits':answer['fresh_fits']},indent=2))


if __name__=='__main__':main()
