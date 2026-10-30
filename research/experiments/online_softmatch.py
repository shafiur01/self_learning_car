"""018: online pseudo-label controls; research-only, no camera activation."""

import argparse
import copy
import hashlib
import json
import math
import platform
import resource
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import evidence_profile as ep
from . import evidence_profile_budget_run as parent
from .run import PROJECT, sha256, write_json

ID = '018_online_softmatch'
PARENT = PROJECT / 'research/experiments/runs/evidence_profile_budget160/20261016T133936682213Z'
OUTPUT = PROJECT / 'research/experiments/runs/online_softmatch'
PLAN = 'research/ONLINE_SOFTMATCH_PROTOCOL.md'
METHODS = ('supervised160', 'online_confidence', 'online_softmatch')
SOURCES = tuple(dict.fromkeys((*parent.SOURCES, PLAN,
    'research/experiments/online_softmatch.py',
    'research/experiments/online_softmatch_audit.py',
    'research/experiments/test_online_softmatch.py')))
METRICS = parent.old.motion.previous.fitting.metric_set


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def half_life_momentum(batch, pool):
    if not 1 < batch <= pool:
        raise ValueError('Require a valid batch and training-pool size.')
    return 2.0 ** (-batch / pool)


class SoftWeights:
    """Paper ordering: raw confidence statistics, UA weights, raw hard target.

    Explicit adaptations: first-batch bootstrap, one-pool-pass half-life, head
    dropout views. Not a bitwise USB or full image-domain reproduction.
    """
    def __init__(self, momentum):
        self.m = momentum
        self.mean = self.var = self.marginal = None

    def update(self, probability):
        p = probability.detach().cpu().numpy().astype(np.float64)
        if p.ndim != 2 or p.shape[1] != 8 or len(p) < 2 or not np.isfinite(p).all():
            raise ValueError('Require finite eight-class probabilities.')
        if np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-6):
            raise ValueError('Invalid probabilities.')
        conf = p.max(1)
        mean, var, marginal = float(conf.mean()), float(conf.var(ddof=1)), p.mean(0)
        if self.mean is None:
            self.mean, self.var, self.marginal = mean, var, marginal
        else:
            self.mean = self.m * self.mean + (1-self.m) * mean
            self.var = self.m * self.var + (1-self.m) * var
            self.marginal = self.m * self.marginal + (1-self.m) * marginal
        aligned = p / np.maximum(self.marginal, 1e-12)
        aligned /= aligned.sum(1, keepdims=True)
        below = np.minimum(aligned.max(1) - self.mean, 0)
        w = np.exp(-2 * below ** 2 / max(self.var, 1e-12))
        target = torch.from_numpy(p.argmax(1)).long()
        weight = torch.from_numpy(w.astype(np.float32))
        return target, weight


def weighted_loss(logits, target, weight):
    if weight.requires_grad or target.requires_grad:
        raise ValueError('Pseudo-label weights and targets must be detached.')
    return (weight * F.cross_entropy(logits, target, reduction='none')).mean()


def stochastic_forward(model, inputs, generator):
    """Private CPU dropout stream; additional passes cannot shift label noise."""
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())
        model.train()
        result = model(inputs)
        generator.set_state(torch.get_rng_state())
    return result


@contextmanager
def payload_guard(allowed, ledger):
    original = np.load
    permitted = {Path(p).resolve() for p in allowed}

    def guarded(path, *args, **kwargs):
        p = Path(path).resolve()
        if p not in permitted:
            raise ValueError(f'Forbidden training payload: {p}')
        ledger.append(str(p))
        return original(path, *args, **kwargs)

    np.load = guarded
    try:
        yield
    finally:
        np.load = original


def preflight():
    m = half_life_momentum(32, 245)
    with torch.random.fork_rng(devices=[]):
        model=ep.Model()
        dropout=[layer for layer in model.modules() if isinstance(layer,torch.nn.Dropout)]
        assert len(dropout)==1 and dropout[0] is model.head[4] and dropout[0].p==.2
        assert model.recurrent.dropout==0
        assert not any(isinstance(layer,torch.nn.modules.batchnorm._BatchNorm) for layer in model.modules())
    # A prediction-only synthetic shift, not a model fit or an oracle-based tune.
    state = SoftWeights(m)
    trace = []
    for step in range(1, 65):
        confidence = .2 if step <= 32 else .9
        p = torch.full((32, 8), (1-confidence)/7, dtype=torch.float64)
        p[:, 0] = confidence
        state.update(p)
        trace.append({'step': step, 'batch_confidence': confidence, 'ema_mean': state.mean})
    expected = .9 + (.2-.9) * m ** 32
    assert abs(state.mean-expected) < 1e-12
    assert abs(m ** (245/32) - .5) < 1e-12
    return {'momentum': m, 'half_life_updates': 245/32,
            'only_stochastic_layer': 'classifier head dropout .2; no sensor transform',
            'original_m999_initial_retention560': .999**560,
            'bootstrap_initial_contribution_after420_updates': m**419,
            'synthetic_trace': trace, 'no_optimization': True}


def prepare():
    config_parent = parent.frozen(PARENT)
    if read(PARENT/'status.json')['phase'] != 'complete':
        raise ValueError('Require completed pilot014.')
    study = OUTPUT / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    study.mkdir(parents=True, exist_ok=False)
    config = {'experiment_id': ID, 'seeds': config_parent['seeds'], 'methods': list(METHODS),
              'steps': 560, 'batch_size': 32, 'learning_rate': .001, 'threads': 2,
              'warmup_steps': 140, 'ramp_steps': 140, 'validation_interval': 20,
              'momentum': half_life_momentum(32,245), 'parameters': ep.PARAMETERS,
              'parent': str(PARENT), 'planned_fits': 15,
              'views': 'identical inputs; deterministic weak, classifier dropout .2 strong',
              'model_ema': False, 'statistics_order': 'raw mean/variance, UA weight, raw target',
              'scope': 'Retrospective development; published-method adaptation, no untouched test.'}
    write_json(study/'config.json', config)
    write_json(study/'preflight.json', preflight())
    for seed in config['seeds']:
        pool = ep.LearnerPool.load(PARENT/f'learner/seed_{seed}.npz')
        assert pool.x.shape == (405,32,146) and len(pool.unlabeled)==245
        np.testing.assert_array_equal(np.bincount(pool.y,minlength=8),[20]*8)
    sources = {name: sha256(PROJECT/name) for name in SOURCES}
    for name in SOURCES:
        p = study/'source'/name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/name,p)
    frozen = {'created_utc': utc(), 'sources': sources,
              'settings': {name: sha256(study/name) for name in ('config.json','preflight.json')},
              'parent': {str(p): sha256(p) for p in PARENT.rglob('*')
                         if p.is_file() and 'plot_cache' not in p.parts},
              'environment': {'python': sys.version, 'numpy': np.__version__,
                              'torch': torch.__version__, 'platform': platform.platform()}}
    write_json(study/'freeze.json', frozen)
    (study/'freeze.sha256').write_text(sha256(study/'freeze.json')+'\n')
    write_json(study/'status.json', {'phase':'prepared','completed_fits':0})
    return study


def frozen(study):
    f = read(study/'freeze.json')
    assert sha256(study/'freeze.json') == (study/'freeze.sha256').read_text().strip()
    for name,h in f['settings'].items():
        assert sha256(study/name)==h, name
    for name,h in f['sources'].items():
        assert sha256(study/'source'/name)==h, name
        if name.endswith('.py'):
            assert sha256(PROJECT/name)==h, name
    for p,h in f['parent'].items():
        assert sha256(Path(p))==h,p
    return read(study/'config.json')


def predict(model,x):
    model.eval()
    with torch.inference_mode():
        return model(ep.observed(x)).softmax(1).numpy()


def save_state(path,state,seed,method,kind,step,study):
    torch.save({'experiment_id':ID,'seed':seed,'method':method,'kind':kind,'step':step,
                'features':149,'parameters':ep.PARAMETERS,'labels':list(parent.LABELS),
                'config_sha256':sha256(study/'config.json'),'state_dict':state},path)


def worker(study,seed,method):
    started = time.perf_counter()
    c = frozen(study)
    if method not in METHODS or seed not in c['seeds']:
        raise ValueError('Undeclared trial.')
    folder=study/f'{method}_seed_{seed}'
    folder.mkdir(exist_ok=False)
    torch.set_num_threads(c['threads'])
    ledger=[]
    allowed=[PARENT/f'learner/seed_{seed}.npz',PARENT/'selection.npz']
    with payload_guard(allowed,ledger):
        pool=ep.LearnerPool.load(allowed[0])
        with np.load(allowed[1],allow_pickle=False) as z:
            vx,vy=z['x'].copy(),z['y'].copy()
        tx,ty=torch.from_numpy(pool.x[pool.labeled]),torch.from_numpy(pool.y)
        ux=torch.from_numpy(pool.x[pool.unlabeled])
        valx,valy=ep.observed(vx),torch.from_numpy(vy)
        fit_seed=seed+300
        torch.manual_seed(fit_seed)
        model=ep.Model()
        initial=ep.state_hash(model)
        label_dropout=torch.Generator()
        label_dropout.set_state(torch.get_rng_state())
        strong_dropout=torch.Generator().manual_seed(fit_seed+5_000_000)
        label_rng=torch.Generator().manual_seed(fit_seed+1_000_000)
        hidden_rng=torch.Generator().manual_seed(fit_seed+3_000_000)
        opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
        class_weight=torch.ones(8)
        statistics=SoftWeights(c['momentum'])
        history,log,weak_trace,weight_trace,row_trace,target_trace=[],[],[],[],[],[]
        hashes={name:hashlib.sha256() for name in ('labeled_draws','hidden_draws','labeled_dropout')}
        best_loss,best_state,best_step=float('inf'),None,None
        warmup_hash=None
        fit_started=time.perf_counter()
        for step in range(1,c['steps']+1):
            li=torch.randint(len(tx),(32,),generator=label_rng)
            hashes['labeled_draws'].update(li.numpy().tobytes())
            hashes['labeled_dropout'].update(label_dropout.get_state().numpy().tobytes())
            opt.zero_grad()
            supervised=F.cross_entropy(stochastic_forward(model,ep.observed(tx[li]),label_dropout),ty[li],weight=class_weight)
            loss=supervised
            record={'step':step,'supervised_loss':float(supervised.detach()),'coefficient':0.,'unsupervised_loss':0.,'mean_weight':0.}
            if method!='supervised160' and step>c['warmup_steps']:
                ui=torch.randint(len(ux),(32,),generator=hidden_rng)
                hashes['hidden_draws'].update(ui.numpy().tobytes())
                batch=ep.observed(ux[ui])
                model.eval()
                with torch.no_grad():
                    probability=model(batch).softmax(1)
                if method=='online_confidence':
                    target=probability.argmax(1)
                    weight=(probability.max(1).values>=.95).float()
                else:
                    target,weight=statistics.update(probability)
                # Same input tensor, only classifier dropout differs in this pass.
                unsupervised=weighted_loss(stochastic_forward(model,batch,strong_dropout),target,weight)
                coefficient=min(1.,(step-c['warmup_steps'])/c['ramp_steps'])
                loss=supervised+coefficient*unsupervised
                weak_trace.append(probability.numpy().copy())
                weight_trace.append(weight.numpy().copy())
                target_trace.append(target.numpy().copy())
                row_trace.append(pool.unlabeled[ui.numpy()])
                record.update({'coefficient':coefficient,'unsupervised_loss':float(unsupervised.detach()),
                               'mean_weight':float(weight.mean()),
                               'ema_mean':statistics.mean,'ema_variance':statistics.var})
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss.')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Non-finite gradient.')
            opt.step()
            log.append(record)
            if step==c['warmup_steps']:
                warmup_hash=ep.state_hash(model)
            if step%c['validation_interval']==0:
                model.eval()
                with torch.inference_mode():
                    ce=F.cross_entropy(model(valx),valy).item()
                history.append({'step':step,'selection_loss':ce})
                if ce<best_loss:
                    best_loss,best_state,best_step=ce,copy.deepcopy(model.state_dict()),step
        fit_seconds=time.perf_counter()-fit_started
        final_state=copy.deepcopy(model.state_dict())
        save_state(folder/'final.pt',final_state,seed,method,'final',560,study)
        save_state(folder/'selected.pt',best_state,seed,method,'selected',best_step,study)
        predictions={}
        for kind,state in (('final',final_state),('selected',best_state)):
            model.load_state_dict(state)
            predictions[kind]=predict(model,vx)
        np.savez_compressed(folder/'development.npz',**predictions)
        if weak_trace:
            np.savez_compressed(folder/'trace.npz',p=np.stack(weak_trace),w=np.stack(weight_trace),
                                target=np.stack(target_trace),rows=np.stack(row_trace))
        summary={'seed':seed,'method':method,'initial_hash':initial,'warmup_hash':warmup_hash,
                 **{name:h.hexdigest() for name,h in hashes.items()},
                 'selected_step':best_step,'selected_before_ssl':best_step<=140,
                 'fit_seconds':fit_seconds,'worker_seconds':time.perf_counter()-started,
                 'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)),
                 'training_labels':len(ty),'training_class_counts':np.bincount(pool.y,minlength=8).tolist(),
                 'supervised_forward_samples':560*32,'weak_forward_samples':len(weak_trace)*32,
                 'strong_forward_samples':len(weak_trace)*32,'development_forward_samples':30*48,
                 'validation_checks':len(history),'optimizer_steps':560,
                 'development':{kind:METRICS(p,vy) for kind,p in predictions.items()}}
        write_json(folder/'history.json',history)
        write_json(folder/'training_log.json',log)
        write_json(folder/'metrics.json',summary)
        write_json(folder/'access.json',{'allowed':[str(p) for p in allowed],'loaded':ledger,
                                       'no_hidden_labels_or_endpoint':True})
    print(f'{method} seed{seed}: selected step{best_step}, devF1={summary["development"]["selected"]["macro_f1"]:.4f}',flush=True)


def train(study):
    c=frozen(study)
    assert read(study/'status.json')=={'phase':'prepared','completed_fits':0}
    started=time.perf_counter()
    trials=[]
    for seed in c['seeds']:
        for method in c['methods']:
            t=time.perf_counter()
            subprocess.run([sys.executable,'-m','research.experiments.online_softmatch','worker',
                            str(study),'--seed',str(seed),'--method',method],cwd=PROJECT,check=True)
            row=read(study/f'{method}_seed_{seed}'/'metrics.json')
            row['subprocess_wall_seconds']=time.perf_counter()-t
            trials.append(row)
            write_json(study/'trials.json',trials)
            write_json(study/'status.json',{'phase':'fitting','completed_fits':len(trials)})
            if method=='supervised160':
                old=read(PARENT/f'supervised160_seed_{seed}/metrics.json')
                assert row['initial_hash']==old['initial_state_sha256']
                assert row['labeled_draws']==old['batch_indices_sha256']
                checkpoint=torch.load(study/f'{method}_seed_{seed}'/'selected.pt',map_location='cpu',weights_only=True)
                control=ep.Model();control.load_state_dict(checkpoint['state_dict'])
                assert ep.state_hash(control)==old['state_sha256'],'Supervised control failed; stop before more fits.'
            paired=[r for r in trials if r['seed']==seed]
            for key in ('initial_hash','warmup_hash','labeled_draws','labeled_dropout'):
                assert len({r[key] for r in paired})==1,(seed,key)
    paths=[p for p in study.iterdir() if p.is_dir() and '_seed_' in p.name]
    files=[p for d in paths for p in d.iterdir() if p.is_file()]
    files.append(study/'trials.json')
    write_json(study/'fit_freeze.json',{'created_utc':utc(),'fits':15,
               'elapsed_subprocess_seconds':time.perf_counter()-started,
               'files':{str(p.relative_to(study)):sha256(p) for p in files}})
    write_json(study/'status.json',{'phase':'fits_complete','completed_fits':15})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','train','worker','analyze','verify'))
    p.add_argument('study',type=Path,nargs='?')
    p.add_argument('--seed',type=int);p.add_argument('--method')
    args=p.parse_args()
    if args.action=='prepare':
        print(prepare());return
    if args.study is None:
        p.error('Provide frozen study path.')
    if args.action=='worker':worker(args.study,args.seed,args.method)
    elif args.action=='train':train(args.study)
    else:
        from . import online_softmatch_audit as audit
        print(json.dumps(getattr(audit,args.action)(args.study),indent=2))


if __name__=='__main__':main()
