"""Experiment016: training-only feature scales and explicit-clock factorial."""

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import evidence_profile as ep
from . import evidence_profile_budget_run as parent
from . import evidence_profile_diagnostic as prior_diagnostic
from .run import PROJECT, sha256, write_json

ID = '016_input_conditioning'
PARENT = prior_diagnostic.PARENT
DIAGNOSTIC = PROJECT / 'research/experiments/runs/evidence_profile_diagnostic/20261016T142255449351Z'
OUTPUT = PROJECT / 'research/experiments/runs/input_conditioning'
CONDITIONS = ('raw_clock', 'rms_clock', 'raw_no_clock', 'rms_no_clock')
ROLES = ('teacher_a', 'teacher_b', 'student')
# name, channels, validity column. Scalar scaling preserves XYZ/XY directions.
BLOCKS = (('pose_xyz', tuple(range(63)), 143), ('wrist_xy', (63,64),143),
          ('log_scale',(65,),143), ('pose_velocity_xyz',tuple(range(67,130)),133),
          ('wrist_velocity_xy',(130,131),133), ('scale_velocity',(132,),133),
          ('orientation',tuple(range(134,143)),143))
SOURCES = tuple(dict.fromkeys((*prior_diagnostic.SOURCES,
    'research/experiments/input_conditioning.py', 'research/experiments/input_conditioning_audit.py',
    'research/experiments/test_input_conditioning.py', 'research/INPUT_CONDITIONING_PROTOCOL.md')))


def read(path):
    return json.loads(Path(path).read_text())


def fit_scales(x):
    """Accept only a caller's training slice; never receive held-out arrays."""
    if x.ndim!=3 or x.shape[1:]!=(32,146) or not np.isfinite(x).all():
        raise ValueError('Require finite training clips of32x146.')
    groups={}
    for name, channels, validity in BLOCKS:
        clip_means=[]
        for clip in x:
            valid=clip[:,validity]>.5
            if valid.any():
                values=clip[valid][:,channels].astype(np.float64)
                clip_means.append(float(np.mean(values**2)))
        rms=float(np.sqrt(np.mean(clip_means))) if clip_means else None
        groups[name]={'rms':rms,'divisor':max(.05,rms) if rms is not None else 1.,
                      'usable_clips':len(clip_means)}
    return {'groups':groups,'training_clips':len(x),'floor':.05,'centering':False}


def role_indices(pool,role):
    return {'teacher_a':pool.teacher_a,'teacher_b':pool.teacher_b,'student':pool.labeled}[role]


def role_scales(pool,role):
    indices=role_indices(pool,role)
    return {**fit_scales(pool.x[indices]),'fit_rows':indices.tolist(),'role':role}


def transform(x,stats,condition):
    if condition not in CONDITIONS:
        raise ValueError('Unknown conditioning condition.')
    result=x.copy()
    if condition.startswith('rms_'):
        for name,channels,_ in BLOCKS:
            result[:,:,list(channels)]/=stats['groups'][name]['divisor']
    if condition.endswith('no_clock'):
        result[:,:,144:146]=0
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite conditioned input.')
    return result


def gate_values(model,x):
    """Reconstruct one-layer LSTM gates from its observed hidden sequence."""
    model.eval()
    with torch.inference_mode():
        observed=ep.observed(x)
        sequence,_=model.recurrent(observed)
        previous=torch.cat((torch.zeros_like(sequence[:,:1]),sequence[:,:-1]),dim=1)
        r=model.recurrent
        z=F.linear(observed,r.weight_ih_l0,r.bias_ih_l0)+F.linear(previous,r.weight_hh_l0,r.bias_hh_l0)
    return z.reshape(len(x),32,4,47).numpy(),sequence.numpy()


def gate_stats(model,x):
    z,_=gate_values(model,x)
    result={}
    for i,name in enumerate(('input','forget','candidate','output')):
        threshold=np.arctanh(.99) if name=='candidate' else math.log(99)
        result[name]={'saturation_fraction':float(np.mean(np.abs(z[:,:,i])>threshold)),
                      'mean_abs_preactivation':float(np.mean(np.abs(z[:,:,i])))}
    return result


def prepare():
    old=parent.frozen(PARENT)
    prior_diagnostic.frozen(DIAGNOSTIC)
    if read(PARENT/'status.json')['phase']!='complete' or read(DIAGNOSTIC/'status.json')['phase']!='complete':
        raise ValueError('Require both completed parent studies.')
    study=OUTPUT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (study/'normalizers').mkdir(parents=True,exist_ok=False)
    config={'experiment_id':ID,'conditions':list(CONDITIONS),'roles':list(ROLES),
            'seeds':old['seeds'],'training':old['training'],'threads':old['threads'],
            'parent_study':str(PARENT),'prior_diagnostic':str(DIAGNOSTIC),
            'parameters':ep.PARAMETERS,'features':149,'prepared_fits':60,
            'scope':'Fixed-budget single-user retrospective baseline experiment; not physical-viewpoint, novelty or independent test evidence.'}
    write_json(study/'config.json',config)
    for seed in config['seeds']:
        pool=ep.LearnerPool.load(PARENT/f'learner/seed_{seed}.npz')
        for role in ROLES:
            write_json(study/'normalizers'/f'{role}_{seed}.json',role_scales(pool,role))
    sources={name:sha256(PROJECT/name) for name in SOURCES}
    for name in SOURCES:
        p=study/'source'/name;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(PROJECT/name,p)
    inputs={str(p.relative_to(study)):sha256(p) for p in [study/'config.json',*(study/'normalizers').iterdir()]}
    freeze={'created_utc':datetime.now(timezone.utc).isoformat(),'inputs':inputs,'sources':sources,
            'parent_artifacts':{str(p):sha256(p) for root in (PARENT,DIAGNOSTIC) for p in root.rglob('*')
                                if p.is_file() and 'plot_cache' not in p.parts}}
    write_json(study/'freeze.json',freeze)
    (study/'freeze.sha256').write_text(sha256(study/'freeze.json')+'\n')
    write_json(study/'trials.json',[]);write_json(study/'status.json',{'phase':'prepared','completed_fits':0})
    write_json(OUTPUT/'latest.json',{'study':str(study)})
    return study


def frozen(study):
    study=Path(study)
    if sha256(study/'freeze.json')!=(study/'freeze.sha256').read_text().strip():
        raise ValueError('Frozen policy changed.')
    f=read(study/'freeze.json')
    if any(sha256(study/n)!=v for n,v in f['inputs'].items()):
        raise ValueError('Prepared settings/scales changed.')
    if any(sha256(Path(n))!=v for n,v in f['parent_artifacts'].items()):
        raise ValueError('A parent artifact changed.')
    for name,digest in f['sources'].items():
        if sha256(study/'source'/name)!=digest or (name.endswith('.py') and sha256(PROJECT/name)!=digest):
            raise ValueError('Frozen executable source changed.')
    return read(study/'config.json')


def load_model(path):
    c=torch.load(path,map_location='cpu',weights_only=True)
    expected={'experiment_id':ID,'features':149,'parameters':ep.PARAMETERS,'labels':list(parent.LABELS)}
    if any(c.get(k)!=v for k,v in expected.items()) or c.get('condition') not in CONDITIONS or c.get('role') not in ROLES:
        raise ValueError('Incompatible research-only conditioning checkpoint.')
    m=ep.Model();m.load_state_dict(c['state_dict']);m.eval()
    return m,c


def train(study):
    study=Path(study);config=frozen(study)
    if read(study/'status.json')['phase']!='prepared' or read(study/'trials.json'):
        raise ValueError('Preserve completed/partial fitting; do not restart.')
    torch.set_num_threads(config['threads'])
    with np.load(PARENT/'selection.npz',allow_pickle=False) as a:
        vx,vy=a['x'].copy(),a['y'].copy()
    trials=[]
    for seed in config['seeds']:
        pool=ep.LearnerPool.load(PARENT/f'learner/seed_{seed}.npz')
        for role,offset in zip(ROLES,(100,200,300)):
            indices=role_indices(pool,role)
            stats_path=study/'normalizers'/f'{role}_{seed}.json'
            scales=read(stats_path)
            if scales!=role_scales(pool,role):
                raise ValueError('Scales do not reproduce from own labeled training rows.')
            y=pool.labels_for(indices)
            for condition in CONDITIONS:
                tx=transform(pool.x[indices],scales,condition);cv=transform(vx,scales,condition)
                model,history,stats=ep.fit(tx,y,cv,vy,seed+offset,config['training'],subset=role!='student')
                p=ep.probabilities(ep.infer(model,cv)[0][0])
                name=f'{condition}_{role}_seed_{seed}';folder=study/name;folder.mkdir(exist_ok=False)
                row={'folder':name,'seed':seed,'role':role,'condition':condition,**stats,
                     'validation':parent.old.motion.previous.fitting.metric_set(p,vy),
                     'normalizer_sha256':sha256(stats_path),'state_sha256':ep.state_hash(model)}
                torch.save({'experiment_id':ID,'features':149,'parameters':ep.PARAMETERS,'labels':list(parent.LABELS),
                            'role':role,'condition':condition,'seed':seed,'best_step':stats['best_step'],
                            'normalizer_sha256':sha256(stats_path),'config_sha256':sha256(study/'config.json'),
                            'freeze_sha256':sha256(study/'freeze.json'),'state_dict':model.state_dict()},folder/'model.pt')
                row['checkpoint_sha256']=sha256(folder/'model.pt')
                write_json(folder/'history.json',history);write_json(folder/'metrics.json',row)
                write_json(folder/'validation_probabilities.json',p.tolist())
                trials.append(row);write_json(study/'trials.json',trials)
                write_json(study/'status.json',{'phase':'fitting','completed_fits':len(trials)})
                print(f"{len(trials)}/60 {name}: dev {row['validation']['accuracy']:.2%}",flush=True)
    # Exact reproduction isolates new conditions from accidental fitter changes.
    previous=read(PARENT/'trials.json')
    for row in trials:
        if row['condition']!='raw_clock':continue
        method='supervised160' if row['role']=='student' else 'teacher_subset'
        other=next(t for t in previous if t['seed']==row['seed'] and t['method']==method
                   and (row['role']=='student' or t['side']==row['role'][-1]))
        if row['state_sha256']!=other['state_sha256']:
            raise ValueError('Raw control did not reproduce parent state exactly.')
    write_json(study/'fit_freeze.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
               'trials_sha256':sha256(study/'trials.json'),
               'checkpoints':{r['folder']:r['checkpoint_sha256'] for r in trials},
               'raw_parent_states_reproduced':15})
    write_json(study/'status.json',{'phase':'fits_complete','completed_fits':60})


def evaluate(study):
    study=Path(study);config=frozen(study)
    if read(study/'status.json')!={'phase':'fits_complete','completed_fits':60}:
        raise ValueError('Complete/freeze all60 fits before retrospective evaluation.')
    if (study/'evaluation_access.json').exists():
        raise ValueError('Preserve inspected evaluation; use verify, not restart.')
    f=read(study/'fit_freeze.json')
    if sha256(study/'trials.json')!=f['trials_sha256'] or any(sha256(study/n/'model.pt')!=v for n,v in f['checkpoints'].items()):
        raise ValueError('Frozen candidates changed.')
    write_json(study/'evaluation_access.json',{'state':'started','created_utc':datetime.now(timezone.utc).isoformat(),'scope':config['scope']})
    out=study/'evaluation';out.mkdir(exist_ok=False)
    try:
        torch.set_num_threads(config['threads'])
        with np.load(PARENT/'endpoint/features.npz',allow_pickle=False) as a:
            ex,ey=a['x'].copy(),a['y'].copy()
        with np.load(PARENT/'oracle.npz',allow_pickle=False) as a:oracle=a['y'].copy()
        endpoint_rows=read(PARENT/'endpoint/manifest.json')['records']
        trials=read(study/'trials.json');results=[]
        for seed in config['seeds']:
            pool=ep.LearnerPool.load(PARENT/f'learner/seed_{seed}.npz')
            for condition in CONDITIONS:
                predictions={};gates={}
                for role in ROLES:
                    row=next(t for t in trials if t['seed']==seed and t['role']==role and t['condition']==condition)
                    model,_=load_model(study/row['folder']/'model.pt')
                    scales=read(study/'normalizers'/f'{role}_{seed}.json')
                    if role=='student':
                        p=ep.probabilities(ep.infer(model,transform(ex,scales,condition))[0][0])
                        predictions['student_endpoint']=p
                    else:
                        anchors=pool.teacher_b if role=='teacher_a' else pool.teacher_a
                        ax=transform(pool.x[anchors],scales,condition)
                        predictions[role+'_anchor']=ep.probabilities(ep.infer(model,ax)[0][0])
                        predictions[role+'_hidden']=ep.probabilities(ep.infer(model,transform(pool.x[pool.unlabeled],scales,condition))[0][0])
                        gates[role]=gate_stats(model,ax)
                pa,pb=predictions['teacher_a_hidden'],predictions['teacher_b_hidden']
                q=(pa+pb)/2;uy=oracle[pool.unlabeled]
                admitted=(pa.argmax(1)==pb.argmax(1))&(pa.max(1)>=.95)&(pb.max(1)>=.95)
                ay=np.r_[pool.labels_for(pool.teacher_b),pool.labels_for(pool.teacher_a)]
                ap=np.r_[predictions['teacher_a_anchor'],predictions['teacher_b_anchor']]
                p=predictions['student_endpoint']
                metric=parent.old.motion.previous.fitting.metric_set
                result={'seed':seed,'condition':condition,'student_endpoint':metric(p,ey),
                        'sittings':{str(s):metric(p[[i for i,r in enumerate(endpoint_rows) if r['sitting']==s]],ey[[i for i,r in enumerate(endpoint_rows) if r['sitting']==s]]) for s in (1,2)},
                        'teacher_cross_session_anchors':metric(ap,ay),'teacher_hidden_mean':metric(q,uy),
                        'admitted_count':int(admitted.sum()),'admitted_errors':int(np.sum(q.argmax(1)[admitted]!=uy[admitted])),
                        'admitted_predicted_counts':{label:int(np.sum(admitted&(q.argmax(1)==c))) for c,label in enumerate(parent.LABELS)},
                        'admitted_actual_counts':{label:int(np.sum(admitted&(uy==c))) for c,label in enumerate(parent.LABELS)},
                        'gates':gates}
                np.savez_compressed(out/f'{condition}_seed_{seed}.npz',**predictions)
                results.append(result)
        write_json(out/'results.json',results)
        write_json(study/'evaluation_access.json',{'state':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
                   'student_models':20,'teacher_models':40,'scope':config['scope']})
        write_json(study/'status.json',{'phase':'complete','completed_fits':60})
    except BaseException as error:
        write_json(study/'evaluation_access.json',{'state':'failed_inspected','error':str(error)});raise
    return {'student_models':20,'teacher_models':40,'no_activation':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','train','evaluate'))
    parser.add_argument('study',type=Path,nargs='?');args=parser.parse_args()
    if args.action=='prepare':print(prepare())
    else:
        if args.study is None:parser.error('Provide the frozen study path.')
        result={'train':train,'evaluate':evaluate}[args.action](args.study)
        if result:print(json.dumps(result,indent=2))


if __name__=='__main__':main()
