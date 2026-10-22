"""Summarize completed short correctness runs; never choose a semantic setting."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.check_semantic_a07_v2 import check, rows


def summarize(root):
    result={"checks":{},"runs":{},"interpretation":"Software correctness only; no demonstrated utility or A07 performance benefit"}
    for budget in (2,20):
        result['checks'][str(budget)]=check(root,budget)
        for mode in ('reference','neutral','active'):
            path=root/'a07'/f'{mode}_{budget}_seed1'
            manifest=json.loads((path/'v2_manifest.json').read_text())
            item=dict(commit=manifest['commit'],config_sha256=manifest['config_sha256'],
                partition_sha256=manifest['partition_sha256'],argv=manifest['argv'],
                resources=json.loads((path/'resources.json').read_text()))
            if mode!='reference':
                records=rows(path/'semantic_v2_rounds.jsonl')
                observations=[o for r in records for o in r['clients'] if o]
                changed=[r for r in records if r['fusion']['changed_parameters']>0]
                regions=[d for r in records for d in r['fusion']['regions'] if d['reason']=='semantic_source_blend']
                share_shift=[abs(d['pi_new'][g]-v) for d in regions for g,v in d['pi_a07'].items()]
                probe_sizes={o['client']:o['probe_examples'] for o in observations}
                item.update(decision_reasons=dict(Counter(r['decision']['reason'] for r in records)),
                    valid_evidence_rounds=[r['round'] for r in records if r['decision']['valid']],
                    nonneutral_rounds=[r['round'] for r in changed],
                    maximum_changed_parameters=max(r['fusion']['changed_parameters'] for r in records),
                    maximum_difference_from_same_input_a07_l2=max(r['fusion']['difference_from_a07_l2'] for r in records),
                    maximum_tier_share_absolute_change=max(share_shift,default=0.),
                    mean_common_valid_classes=mean(len(r['decision']['classes']) for r in records),
                    maximum_common_valid_classes=max(len(r['decision']['classes']) for r in records),
                    backbone_forward_examples=sum(o['backbone_forward_examples'] for o in observations),
                    adapter_backward_examples=sum(o['adapter_backward_examples'] for o in observations),
                    measurement_seconds=sum(o['pre_measurement_seconds']+o['post_measurement_seconds'] for o in observations),
                    local_train_seconds=sum(o['native_local_train_seconds'] for o in observations),
                    memory_commit_seconds=sum(r['memory_commit_seconds'] for r in records),
                    score_seconds=sum(r['decision']['score_seconds'] for r in records),
                    fusion_seconds=sum(r['fusion']['fusion_seconds'] for r in records),
                    auxiliary_logical_upload_bytes=sum(o['logical_upload_bytes'] for o in observations),
                    decision_message_json_bytes=sum(r['decision_message_json_bytes'] for r in records),
                    final_adapter_tensor_bytes=records[-1]['adapter_storage_bytes'],
                    fixed_probe_image_label_tensor_bytes=sum(probe_sizes.values())*(3*32*32*4+8),
                    final_tier_anchor_json_bytes=records[-1]['tier_anchor_json_bytes'],
                    final_tier_reliability_json_bytes=records[-1]['tier_reliability_json_bytes'],
                    head_sha256={seed:sorted({o['heads'][seed]['head_sha256'] for o in observations})
                        for seed in observations[0]['heads']} if observations else {},
                    adapter_in_decision=False)
            result['runs'][f'{mode}_{budget}']=item
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    opt=p.parse_args();report=summarize(opt.root);opt.output.write_text(json.dumps(report,indent=2));print(opt.output)
