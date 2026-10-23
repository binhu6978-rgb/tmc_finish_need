"""Replay past-only V2 decisions and describe real500 evidence, never train or score-search."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from semcons_a07_v2.memory import TierSemanticMemory,TierReliabilityMemory,tier_of
from semcons_a07_v2.scoring import complementarity
from semcons_a07_v2.aggregation import blend_shares
from tools.analyze_semantic_a07_iv_e import mean,corr,interval,cluster_weights,points_from,associations,write_csv


def canon(value):return json.loads(json.dumps(value))


def audit(root,output):
    index=json.loads((root/'scalar_evidence_index.json').read_text())
    for name,record in index.items():
        p=root/name
        assert p.stat().st_size==record['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
    result=dict(original_scalar_files_verified=len(index),trajectories={})
    protocol=json.loads((ROOT/'configs/semantic_a07_pair500.json').read_text())
    for mode in ('neutral','active'):
        path=root/mode/'training/a07'/f'{mode}_500_seed1'
        manifest=json.loads((root/mode/'manifest.json').read_text());cfg=manifest['identity']['cfg']
        semantic=TierSemanticMemory(cfg);reliability=TierReliabilityMemory(cfg)
        logs=[json.loads(l) for l in (path/'semantic_v2_rounds.jsonl').read_text().splitlines()]
        regression=[json.loads(l) for l in (path/'regression.jsonl').read_text().splitlines()]
        cost=defaultdict(float);classes=Counter();source_count=Counter();rhos=[];us=[];max_share=0.;modified=[]
        for row in logs:
            rd=row['round'];current={(v['tier'],v['category']):{k:w for k,w in v.items() if k not in ('tier','category')} for v in row['current_task']}
            decision=complementarity(current,semantic.tier_snapshot(),reliability.tier_snapshot(),reliability,
                rd,{tier_of(c,(40,70)) for c,p in regression[rd]['slots']},cfg)
            saved={k:v for k,v in row['decision'].items() if k!='score_seconds'}
            assert canon(decision)==saved, f'Past-only score replay mismatch round{rd}'
            updates=semantic.commit_tiers(rd,current)
            assert canon([dict(tier=g,category=c,**v) for (g,c),v in sorted(updates.items())])==row['tier_commits']
            reliability.commit_tiers(rd,updates)
            for detail in saved['details']:
                assert detail['reference_round']<rd
                rhos.append(detail['rho']);us.append(detail['u'])
            classes[len(saved['classes'])]+=1
            for g in saved['scores']:source_count[int(g)]+=1
            report=row['fusion'];modified.append(report['difference_from_a07_l2'])
            for region in report['regions']:
                if 'pi_new' not in region:continue
                pi={int(g):v for g,v in region['pi_a07'].items()}
                scores={int(g):v for g,v in saved['scores'].items()}
                expected=blend_shares(pi,scores,report['lambda_value'])
                assert canon(expected)==region['pi_new']
                assert all(v>=0 and np.isfinite(v) for v in expected.values()) and abs(sum(expected.values())-1)<1e-12
                max_share=max(max_share,max(abs(expected[g]-pi[g]) for g in pi))
            for obs in row['clients']:
                if obs is None:continue
                for key in ('pre_measurement_seconds','post_measurement_seconds','native_local_train_seconds',
                    'probe_examples','backbone_forward_examples','adapter_backward_examples','logical_upload_bytes'):
                    cost[key]+=obs.get(key,0)
            for key in ('decision_message_json_bytes','memory_commit_seconds'):
                cost[key]+=row[key]
        r=[json.loads(l) for l in (path/'iv_e_rounds.jsonl').read_text().splitlines()]
        f7=[v for rd in r for v in rd['figure7']]
        f8=[v|rd['norms'][v['method']]|dict(active=rd['norms']['V2']['difference_from_a07_l2']>0) for rd in r for v in rd['figure8']]
        differences=[];rng=np.random.default_rng(protocol['analysis_seed']);comparisons={}
        primary=[v for v in f8 if v['cohort']=='confirmation' and v['active']]
        v2={(v['round'],v['receiver']):v for v in primary if v['method']=='V2'}
        for method in ('no_sm','no_rm','permuted','A07_L2_matched'):
            paired=[dict(v,utility_vs_a07=v2[v['round'],v['receiver']]['utility_vs_a07']-v['utility_vs_a07']) for v in primary if v['method']==method]
            draws=[]
            for _ in range(protocol['bootstrap_repetitions']):
                w=cluster_weights(paired,rng)
                if len(w) and w.sum():draws.append(float(np.average([p['utility_vs_a07'] for p in paired],weights=w)))
            comparisons[method]=dict(mean_v2_minus_control_utility=mean([p['utility_vs_a07'] for p in paired]),cluster95=interval(draws))
        shared=[v for v in f7 if v['cohort']=='confirmation' and v['U'] is not None and v['common_utility'] is not None]
        # Class-composition decomposition uses the identical injection and probe batches.
        decomposition={}
        for label,items in [('common',shared),('noncommon',[v for v in shared if v['noncommon_utility'] is not None])]:
            pts=points_from([dict(v,common_utility=v['common_utility'] if label=='common' else v['noncommon_utility']) for v in items])
            decomposition[label]=dict(units=len(pts),mean_utility=mean([p['utility'] for p in pts]),
                U_spearman=corr([p['U'] for p in pts],[p['utility'] for p in pts],spearman=True))
        contrast=[]
        for rd in r:
            n=rd['norms'];base=n['A07']['update_l2'];active=rd['norms']['V2']['difference_from_a07_l2']>0
            for method in ('V2','permuted','no_sm','no_rm','A07_L2_matched'):
                contrast.append(dict(round=rd['round'],method=method,active=active,update_l2=n[method]['update_l2'],
                    update_l2_ratio_a07=n[method]['update_l2']/base if base else None,
                    difference_l2=n[method]['difference_from_a07_l2'],cosine_a07=n[method]['update_cosine_a07']))
        write_csv(output/mode/'update_scales.csv',contrast)
        actnorms=[v for v in contrast if v['active'] and v['method']=='V2']
        result['trajectories'][mode]=dict(past_only_score_replay_rounds=len(logs),replayed_tier_commits=True,
            source_weight_formula_replayed=True,primary_paired_control_comparisons=comparisons,
            primary_class_decomposition=decomposition,current_class_eligibility_counts=dict(classes),
            eligible_rounds_per_tier=dict(source_count),rho_summary=dict(min=min(rhos),mean=mean(rhos),max=max(rhos)),
            category_u_summary=dict(min=min(us),mean=mean(us),max=max(us)),
            maximum_absolute_regional_share_change=max_share,
            applied_modified_l2_mean_all_rounds=mean(modified),applied_modified_l2_max=max(modified),
            sampled_v2_active_update_l2_ratio_a07_mean=mean([v['update_l2_ratio_a07'] for v in actnorms]),
            sampled_v2_active_update_cosine_a07_mean=mean([v['cosine_a07'] for v in actnorms]),
            sampled_v2_active_difference_l2_mean=mean([v['difference_l2'] for v in actnorms]),
            diagnostic_seconds=sum(v['seconds'] for v in r),diagnostic_forward_examples=sum(v['receiver_forward_examples'] for v in r),
            hypothetical_receiver_feedback_bytes=sum(v['hypothetical_class_feedback_bytes'] for v in r),
            hypothetical_region_downlink_bytes=sum(v['hypothetical_tier_delta_downlink_bytes'] for v in r),
            costs=dict(cost),final_storage=dict(adapter_bytes=logs[-1]['adapter_storage_bytes'],
                tier_anchor_json_bytes=logs[-1]['tier_anchor_json_bytes'],tier_history_json_bytes=logs[-1]['tier_reliability_json_bytes']),
            training_manifest_commit=manifest['commit'],configuration_sha256=manifest['configuration_sha256'],
            partition_sha256=manifest['identity']['partition'])
    (output/'offline_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    opt=p.parse_args();audit(opt.root,opt.output)
