"""Read-only mechanism journal and bounded round-boundary recovery for frozen V2."""
import copy
from collections import defaultdict
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import torch
import adapterfl_methods as methods
from models.Fed import split_model
from semcons_a07.diagnostics import append_json, rng_digest
from semcons_a07.trainer import OriginalHooks
from .aggregation import fuse
from .iv_e_diagnostics import independent_indices,Probes
from semcons_a07.measurement import materialize
from semcons_a07.config import seed_of
from .iv_e_holdout import HeldoutDiagnostic
from .trainer import semantic_loop


def sampled_rounds(protocol):
    return sorted(set(protocol['diagnostic_rounds']) | {
        r for a,b in protocol['stages'].values() for r in range(a,b+1,protocol['window_stride'])})


def evaluation_indices(dataset,users):
    return {c:independent_indices(dataset,indices,c,classes=100,samples=2) for c,indices in users.items()}


class TrainingSideDiagnostic(HeldoutDiagnostic):
    """Shares heldout tooling only; all these probe samples remain in original SGD."""
    def __init__(self,dataset,users,evaluation,path):
        super().__init__(dataset,users,evaluation,path)
        class EvaluationProbes(Probes):
            def batches(self,cid):
                if cid not in self.cache:
                    self.cache[cid]=materialize(dataset,evaluation[cid],seed_of('pair500-views',cid),50)
                return self.cache[cid]
        self.probes=EvaluationProbes(dataset,users)

    def pairs(self,*args,**kwargs):
        rows,inventory=super().pairs(*args,**kwargs)
        for row in rows:
            row['source_own_evaluation_gain']=row.pop('source_own_holdout_gain')
            row['probe_scope']='SGD training-side, disjoint from semantic measurement'
        return rows,inventory


def recovery_snapshot(log,observer,models,identity):
    if observer.measurement.pending:raise ValueError('Recovery requires committed measurement state')
    return dict(schema=1,identity=identity,next_round=log.next_round,
        full=methods.cpu_copy(models[-1].state_dict()),method=methods.cpu_copy(log.hooks.state_dict()),
        rows=copy.deepcopy(log.rows),total_time=log.total_time,
        elapsed=log.elapsed_before+time.perf_counter()-log.started,
        semantic=methods.cpu_copy(observer.semantic.__dict__),
        reliability=methods.cpu_copy(observer.reliability.__dict__),
        adapter_bank=methods.cpu_copy(observer.measurement.bank),
        head_hashes=copy.deepcopy(observer.measurement.head_hashes),rng=methods.rng_state())


def restore_snapshot(state,log,observer,models,identity):
    if state['schema']!=1 or state['identity']!=identity:raise ValueError('Recovery identity differs')
    n=state['next_round']
    if n!=len(state['rows']) or [r['round'] for r in state['rows']]!=list(range(1,n+1)):
        raise ValueError('Recovery row sequence differs')
    # Frozen Final40 explicitly disallows its old no-PT resume. This independent
    # entry restores its only dynamic state after checking its frozen config.
    h=state['method']
    if h['method']!=log.hooks.method or h['config']!=log.hooks.config or h['rounds_seen']!=n:
        raise ValueError('Frozen A07 hook state differs')
    log.hooks.rounds_seen=n
    for model in models:model.load_state_dict(split_model(state['full'],model.state_dict()))
    log.rows=state['rows'];log.next_round=n;log.total_time=state['total_time'];log.elapsed_before=state['elapsed']
    observer.semantic.__dict__=state['semantic'];observer.reliability.__dict__=state['reliability']
    # cpu_copy serializes defaultdict as dict. Recreate missing-key behavior for
    # future client/profile/category observations after an explicitly approved restore.
    for key in ('clients','profiles','tiers'):
        observer.reliability.__dict__[key]=defaultdict(list,observer.reliability.__dict__[key])
    observer.measurement.bank=state['adapter_bank'];observer.measurement.head_hashes=state['head_hashes']
    observer.round_number=n;methods.restore_rng(state['rng'])


def atomic_checkpoint(path,state):
    temp=path.with_suffix('.pending');torch.save(state,temp);temp.replace(path)


@contextmanager
def pair_loop(cfg,dataset,users,path,mode,boundaries,protocol,identity,diagnostic=True,resume=None):
    rounds=set(sampled_rounds(protocol))
    with semantic_loop(cfg,dataset,users,path,mode,boundaries) as observer:
        old_log=methods.RunLog;diag=TrainingSideDiagnostic(dataset,users,evaluation_indices(dataset,users),path)
        diag.observer=observer
        class Log(old_log):
            def __init__(self,args,models):
                super().__init__(args,models);self.models=models;diag.models=models
                if resume:
                    # This entry restores only an exact aligned journal. It never truncates later evidence.
                    state=torch.load(resume,map_location='cpu',weights_only=False)
                    for handle in (self.accuracy,self.diagnostics):handle.close()
                    source=Path(resume).parent
                    for name in ('accuracy_each_round.csv','method_diagnostics.csv'):
                        with (source/name).open() as f:assert len(list(csv.DictReader(f)))==state['next_round']
                    for name in ('regression.jsonl','semantic_v2_rounds.jsonl','source_weights_each_round.jsonl'):
                        values=[json.loads(l) for l in (source/name).read_text().splitlines()]
                        assert max(v['round'] for v in values)==state['next_round']
                    for name in ('accuracy_each_round.csv','method_diagnostics.csv','regression.jsonl',
                                 'semantic_v2_rounds.jsonl','source_weights_each_round.jsonl','iv_e_rounds.jsonl'):
                        if (source/name).exists():shutil.copyfile(source/name,path/name)
                    restore_snapshot(state,self,observer,models,identity)
                    self.accuracy=(path/'accuracy_each_round.csv').open('a',newline='')
                    self.diagnostics=(path/'method_diagnostics.csv').open('a',newline='')
                    self.accuracy_writer=csv.DictWriter(self.accuracy,fieldnames=list(self.rows[0]))
                    self.diagnostic_writer=csv.DictWriter(self.diagnostics,fieldnames=['round','method','metrics_json'])
                normal=self.hooks.aggregate
                def aggregate(uploads,lens,previous,dispatched=None):
                    shadow=copy.copy(self.hooks);first=shadow.rounds_seen==0
                    actual=normal(uploads,lens,previous,dispatched)
                    base=OriginalHooks.aggregate(shadow,uploads,lens,previous,dispatched)
                    virtual,report=fuse(base,previous,uploads,lens,self.hooks,observer.decision,
                        cfg['semantic_lambda'],first)
                    append_json(path/'source_weights_each_round.jsonl',dict(round=observer.round_number,
                        applied_mode=mode,a07=self.hooks.round_diagnostics['a07_baseline'],
                        virtual_v2=report,applied=observer.fusion))
                    if diagnostic and observer.round_number in rounds:
                        applied_report=observer.fusion;observer.fusion=report
                        try:diag.run(base,virtual,previous,uploads,lens,self.hooks,observer.decision,first)
                        finally:observer.fusion=applied_report
                    return actual
                self.hooks.aggregate=aggregate

            def record(self,round_index,models,evaluations,*args):
                super().record(round_index,models,evaluations,*args)
                if self.next_round%protocol['checkpoint_interval']==0 or self.next_round==self.args.epochs:
                    before=rng_digest();state=recovery_snapshot(self,observer,models,identity)
                    atomic_checkpoint(path/'recovery_latest.pt',state)
                    if rng_digest()!=before:raise AssertionError('Recovery snapshot changed training RNG')
                    (path/'recovery_latest.json').write_text(json.dumps(dict(round=self.next_round,
                        bytes=(path/'recovery_latest.pt').stat().st_size,contains_memories=True,
                        selection='fixed interval/latest, never selected by test'),indent=2))
        try:
            methods.RunLog=Log;yield observer
        finally:methods.RunLog=old_log


def verify_pair(root,n):
    paths={m:root/m/'training/a07'/f'{m}_{n}_seed1' for m in ('neutral','active')}
    records={m:[json.loads(l) for l in (p/'regression.jsonl').read_text().splitlines()] for m,p in paths.items()}
    a,b=records.values();assert len(a)==len(b)==n+1
    assert a[0]==b[0] and a[1]['models']==b[1]['models']
    for x,y in zip(a[1:],b[1:]):
        assert x['round']==y['round'] and x['slots']==y['slots'] and x['rng']==y['rng']
        assert [(c['before_train_rng'],c['after_train_rng']) for c in x['local_checks']]==[(c['before_train_rng'],c['after_train_rng']) for c in y['local_checks']]
    for mode,p in paths.items():
        rows=list(csv.DictReader((p/'accuracy_each_round.csv').open()))
        assert [int(r['round']) for r in rows]==list(range(1,n+1))
        assert all(math.isfinite(float(r[k])) for r in rows for k in ('full_accuracy','full_ce'))
        assert json.loads((p/'completed.json').read_text())['success']
    return dict(success=True,rounds=n,initial_models_equal=True,first_round_models_equal=True,
        all_schedules_and_training_rng_equal=True,models_after_semantic_activation_expected_to_differ=True)
