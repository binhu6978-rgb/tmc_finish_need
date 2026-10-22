"""Scoped V2 hook of the original AdaptiveFL loop; main SGD is not copied."""
import copy
import json
import time
from contextlib import contextmanager
import torch
import adapterfl_methods as methods
import capability_final40 as frozen
import Algorithm.Training_AdaptiveFL as training
from semcons_a07.trainer import Observer,OriginalHooks,OriginalLog,plain_observation
from semcons_a07.diagnostics import append_json,model_digest,rng_digest
from native_group_aggregation import update_norm
from .memory import TierSemanticMemory,TierReliabilityMemory,task_rows,tier_of
from .scoring import complementarity
from .aggregation import fuse


class SemanticObserver(Observer):
    def __init__(self,cfg,dataset,users,path,enabled,boundaries):
        super().__init__(cfg,dataset,users,path,enabled)
        self.semantic=TierSemanticMemory(cfg);self.reliability=TierReliabilityMemory(cfg)
        self.boundaries=boundaries;self.decision=None;self.fusion=None

    def start(self,clients,profiles):
        super().start(clients,profiles)
        self.tier_anchors=self.semantic.tier_snapshot()
        self.tier_history=self.reliability.tier_snapshot()

    def score(self):
        started=time.perf_counter()
        self.current=task_rows(self.observations,self.boundaries,self.cfg['minimum_class_support'])
        tiers={tier_of(c,self.boundaries) for c,p in self.slots}
        self.decision=complementarity(self.current,self.tier_anchors,self.tier_history,self.reliability,
                                     self.round_number,tiers,self.cfg)
        self.decision['score_seconds']=time.perf_counter()-started
        return self.decision

    def commit(self):
        started=time.perf_counter()
        records=[];legacy_history=[]
        for obs in self.observations:
            if obs is None:continue
            cid,profile=obs['client'],obs['profile']
            records.append((cid,'task',obs['task_after']))
            records.extend((cid,seed,head['after']) for seed,head in obs['heads'].items())
            legacy_history.append((cid,profile,obs['heads'][self.cfg['head_seeds'][0]]['evidence']['S1']))
        legacy_semantic=self.semantic.commit(self.round_number,records)
        updates=self.semantic.commit_tiers(self.round_number,self.current)
        self.reliability.commit(self.round_number,legacy_history)
        self.reliability.commit_tiers(self.round_number,updates)
        self.measurement.commit()
        decision_bytes=0
        for obs in self.observations:
            if obs is not None:
                payload=[dict(category=y,q_pre=float(obs['task_before'][y]['probability'][y]),
                    q_post=float(obs['task_after'][y]['probability'][y]),support=obs['task_after'][y]['count'])
                    for y in sorted(obs['task_before'].keys() & obs['task_after'].keys())]
                decision_bytes+=len(json.dumps(payload,separators=(',',':')).encode('utf-8'))
        anchor_bytes=len(json.dumps([dict(tier=g,category=c,**v) for (g,c),v in self.semantic.tiers.items()]).encode('utf-8'))
        history_bytes=len(json.dumps([dict(tier=g,category=c,history=v) for (g,c),v in self.reliability.tiers.items()]).encode('utf-8'))
        append_json(self.path/'semantic_v2_rounds.jsonl',dict(round=self.round_number,
            decision=self.decision,fusion=self.fusion,
            current_task=[dict(tier=g,category=c,**value) for (g,c),value in sorted(self.current.items())],
            tier_commits=[dict(tier=g,category=c,**value) for (g,c),value in sorted(updates.items())],
            legacy_semantic_auxiliary=legacy_semantic,clients=[plain_observation(o) for o in self.observations],
            adapter_role='auxiliary only; excluded from U and all applied source weights',
            adapter_storage_bytes=self.measurement.storage_bytes(),
            decision_message_json_bytes=decision_bytes,tier_anchor_json_bytes=anchor_bytes,
            tier_reliability_json_bytes=history_bytes,memory_commit_seconds=time.perf_counter()-started,
            tier_anchor_count=len(self.semantic.tiers),tier_history_records=sum(map(len,self.reliability.tiers.values()))))


@contextmanager
def semantic_loop(cfg,dataset,users,path,mode,boundaries):
    observer=SemanticObserver(cfg,dataset,users,path,mode!='reference',boundaries)
    old_select=training.select_clients;old_hooks=frozen.Final40Hooks;old_log=methods.RunLog
    strength=cfg['semantic_lambda'] if mode=='active' else 0.

    def select(args,profiles,number):
        clients=old_select(args,profiles,number);observer.start(clients,profiles);return clients

    class Hooks(OriginalHooks):
        def __init__(self,args,models):
            super().__init__(args,models)
            self.config=copy.deepcopy(self.config)
            self.config['semantic_v2']=dict(lambda_value=strength,adapter_in_decision=False,
                reference='past tier/category true-class task probability',bn='unchanged A07',classifier='Native')

        def configure_local(self,local):
            super().configure_local(local);observer.wrap(local)

        def aggregate(self,uploads,lens,previous,dispatched=None):
            first=self.rounds_seen==0
            base=super().aggregate(uploads,lens,previous,dispatched)
            original=copy.deepcopy(self.round_diagnostics)
            decision=observer.score() if observer.enabled else dict(valid=False,reason='reference',scores={},classes=[],details=[])
            begin=time.perf_counter()
            result,report=fuse(base,previous,uploads,lens,self,decision,strength,first)
            report['fusion_seconds']=time.perf_counter()-begin
            observer.fusion=report
            self.round_diagnostics=dict(a07_baseline=original,semantic_v2=report,
                final_update_l2=update_norm(result,previous),a07_update_l2=original['actual_update_l2'])
            return result

    class Log(OriginalLog):
        def __init__(self,args,models):
            super().__init__(args,models)
            append_json(path/'regression.jsonl',dict(round=0,models=[model_digest(m) for m in models],rng=rng_digest()))

        def record(self,round_index,models,evaluations,*args):
            super().record(round_index,models,evaluations,*args)
            if observer.enabled:observer.commit()
            append_json(path/'regression.jsonl',dict(round=round_index+1,models=[model_digest(m) for m in models],
                rng=rng_digest(),evaluations=evaluations,slots=observer.slots,local_checks=observer.local_checks))

    try:
        training.select_clients=select;frozen.Final40Hooks=Hooks;methods.RunLog=Log
        yield observer
    finally:
        training.select_clients=old_select;frozen.Final40Hooks=old_hooks;methods.RunLog=old_log
