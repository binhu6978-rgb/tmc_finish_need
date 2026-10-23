"""Read-only completion, memory replay and recovery audit of the single1000 run."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from semcons_a07_v2.memory import TierSemanticMemory,TierReliabilityMemory,tier_of
from semcons_a07_v2.scoring import complementarity
from semcons_a07_v2.aggregation import blend_shares
from tools.report_semantic_activation import write_csv


def load(path):return json.loads(path.read_text())
def rows(path):return [json.loads(x) for x in path.read_text().splitlines()]
def canonical(x):return json.loads(json.dumps(x))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root,reference):
    run=root/'active/training/a07/active_1000_seed1';out=root/'delivery_audit'
    out.mkdir(exist_ok=False)
    manifest=load(root/'active/manifest.json');protocol=manifest['identity']['protocol'];cfg=manifest['identity']['cfg']
    assert protocol['rounds']==1000 and manifest['identity']['epochs']==1000
    assert load(root/'status.json')['success'] and load(run/'completed.json')['rounds_completed']==1000
    metrics=list(csv.DictReader((run/'accuracy_each_round.csv').open()))
    assert len(metrics)==1000 and [int(r['round']) for r in metrics]==list(range(1,1001))
    assert all(math.isfinite(float(r[k])) for r in metrics for k in ('full_accuracy','full_ce'))
    logs=rows(run/'semantic_v2_rounds.jsonl');regression=rows(run/'regression.jsonl')
    weights=rows(run/'source_weights_each_round.jsonl');diagnostics=rows(run/'iv_e_rounds.jsonl')
    assert [r['round'] for r in logs]==[r['round'] for r in weights]==list(range(1,1001))
    assert [r['round'] for r in regression]==list(range(1001))
    expected=sorted(set(protocol['diagnostic_rounds'])|{r for a,b in protocol['stages'].values() for r in range(a,b+1)})
    assert [r['round'] for r in diagnostics]==expected and len(expected)==53
    source={name:digest(ROOT/name)==sha for name,sha in manifest['identity']['source'].items()}
    assert all(source.values())
    semantic=TierSemanticMemory(cfg);reliability=TierReliabilityMemory(cfg);cost=defaultdict(float)
    for r in logs:
        rd=r['round'];current={(v['tier'],v['category']):{k:w for k,w in v.items() if k not in ('tier','category')} for v in r['current_task']}
        decision=complementarity(current,semantic.tier_snapshot(),reliability.tier_snapshot(),reliability,
            rd,{tier_of(c,(40,70)) for c,p in regression[rd]['slots']},cfg)
        assert canonical(decision)=={k:v for k,v in r['decision'].items() if k!='score_seconds'},rd
        updates=semantic.commit_tiers(rd,current)
        assert canonical([dict(tier=g,category=c,**v) for (g,c),v in sorted(updates.items())])==r['tier_commits']
        reliability.commit_tiers(rd,updates)
        for region in r['fusion']['regions']:
            if 'pi_new' not in region:continue
            pi={int(g):v for g,v in region['pi_a07'].items()};scores={int(g):v for g,v in decision['scores'].items()}
            assert canonical(blend_shares(pi,scores,r['fusion']['lambda_value']))==region['pi_new']
        for obs in r['clients']:
            if obs is None:continue
            for k in ('pre_measurement_seconds','post_measurement_seconds','native_local_train_seconds','probe_examples',
                'backbone_forward_examples','adapter_backward_examples','logical_upload_bytes'):cost[k]+=obs.get(k,0)
        for k in ('decision_message_json_bytes','memory_commit_seconds'):cost[k]+=r[k]
    state=torch.load(run/'recovery_latest.pt',map_location='cpu',weights_only=False)
    assert state['identity']==manifest['identity'] and state['next_round']==1000 and len(state['rows'])==1000
    h=hashlib.sha256()
    for k,v in state['full'].items():
        assert bool(torch.isfinite(v).all());h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
    assert h.hexdigest()==regression[-1]['models'][-1]
    assert load(run/'recovery_latest.json')['round']==1000
    adapter_bytes=sum(v.numel()*v.element_size() for bank in state['adapter_bank'].values() for v in bank.values())
    assert adapter_bytes==logs[-1]['adapter_storage_bytes']
    prefix=None
    if reference:
        old=reference/'active/training/a07/active_500_seed1';oldreg=rows(old/'regression.jsonl')
        oldmetrics=list(csv.DictReader((old/'accuracy_each_round.csv').open()))
        prefix=dict(rounds=500,seven_models_exact=all(a['models']==b['models'] for a,b in zip(oldreg,regression)),
            slots_rng_exact=all({k:v for k,v in a.items() if k!='models'}=={k:v for k,v in b.items() if k!='models'} for a,b in zip(oldreg,regression)),
            accuracy_ce_exact=all(a[k]==b[k] for a,b in zip(oldmetrics,metrics) for k in ('full_accuracy','full_ce')))
    f7=[v for d in diagnostics for v in d['figure7']];f8=[v for d in diagnostics for v in d['figure8']]
    write_csv(out/'receiver_utility_raw.csv',f7);write_csv(out/'full_counterfactual_raw.csv',f8)
    stages={}
    for label,(a,b) in protocol['stages'].items():
        subset=[r for r in logs if a<=r['round']<=b]
        stages[label]=dict(rounds=len(subset),active=sum(r['fusion']['difference_from_a07_l2']>0 for r in subset),
            mean_difference_l2=float(np.mean([r['fusion']['difference_from_a07_l2'] for r in subset])),
            max_difference_l2=max(r['fusion']['difference_from_a07_l2'] for r in subset))
    peak=max(metrics,key=lambda r:float(r['full_accuracy']))
    result=dict(rounds=1000,finite_accuracy_ce=True,regression_records=len(regression),semantic_records=len(logs),
        weight_records=len(weights),fixed_diagnostic_rounds=expected,diagnostic_records=len(diagnostics),
        source_files_unchanged=source,past_only_exact_replay_rounds=1000,source_weight_formula_exact=True,
        checkpoint=dict(round=1000,identity_equal=True,finite_full=True,final_journal_sha_equal=True,
            full_sha256=h.hexdigest(),bytes=(run/'recovery_latest.pt').stat().st_size,
            adapter_entries=len(state['adapter_bank']),adapter_bytes=adapter_bytes,rng_streams=sorted(state['rng']),
            note='Saved state verified without resuming training; inherited no-checkpoint text in generic summary refers to disabled generic/best saves, not this explicit recovery writer'),
        prefix500=prefix,stages=stages,costs=dict(cost),resources=load(root/'active/resources.json'),
        final_storage=dict(adapter_bytes=adapter_bytes,tier_anchor_json_bytes=logs[-1]['tier_anchor_json_bytes'],
            tier_history_json_bytes=logs[-1]['tier_reliability_json_bytes']),
        performance=dict(peak_label='test-oracle peak',peak=float(peak['full_accuracy']),peak_round=int(peak['round']),
            final=float(metrics[-1]['full_accuracy']),final_ce=float(metrics[-1]['full_ce']),
            last20=float(np.mean([float(r['full_accuracy']) for r in metrics[-20:]])),
            last50=float(np.mean([float(r['full_accuracy']) for r in metrics[-50:]]))),
        raw_utility_counts=dict(receiver_rows=len(f7),full_rows=len(f8)),
        diagnostic_seconds=sum(r['seconds'] for r in diagnostics),
        diagnostic_forward_examples=sum(r['receiver_forward_examples'] for r in diagnostics),
        no_new_training=True,scope='Operation audit; no inference of semantic benefit from activation')
    (out/'completion_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    # Full journals remain on the server, indexed for reproducibility; no weights or private examples in archive.
    paths=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.log','.png','.pdf','.md')]
    index={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(paths)}
    (out/'original_file_index.json').write_text(json.dumps(index,indent=2))
    selected=[p for p in paths if ('activation_report' in p.parts or 'delivery_audit' in p.parts or
        p.stat().st_size<1000000 or p.name=='regression.jsonl')]
    archive=root/'delivery_scalar_evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in [*selected,out/'original_file_index.json']:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
    print(json.dumps(dict(audit=result,archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);p.add_argument('--reference500',type=Path)
    opt=p.parse_args();audit(opt.root,opt.reference500)
