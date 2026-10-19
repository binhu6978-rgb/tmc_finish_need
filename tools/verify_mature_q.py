"""Offline provenance / scalar replay of the fixed mature uploads; no SGD or test data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from semcons_scope import ScopeHooks, semantic_evidence
from tools.probe_semcons_directions import references, signed_cos
from tools.mature_receiver_ce import sha, state_sha


def verify(root, checkpoint_path, partition_path):
    protocol=json.loads((root/'protocol.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    assert sha(checkpoint_path)==protocol['checkpoint_sha256']==manifest['checkpoint_sha256']
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    anchor=checkpoint['global']; assert state_sha(anchor)==manifest['anchor_sha256']
    users=json.loads(partition_path.read_text())['train_data']
    assert sha(partition_path)==protocol['partition_sha256']
    batches=[json.loads(x) for x in (root/'receiver_ce.jsonl').read_text().splitlines()]
    assert len(batches)==5 and all(b['anchor_rng_history_unchanged'] for b in batches)
    plans=json.loads((root/'probe_plans.json').read_text())
    probes=torch.load(root/'fixed_probes.pt',map_location='cpu',weights_only=False)
    for cid,plan in plans.items():
        assert set(plan['indices'])<=set(users[cid]) and len(plan['indices'])<=32
        assert plan['batch_sha256']==probes[int(cid)]['sha256']
        probe=probes[int(cid)]
        digest=hashlib.sha256(probe['x'].contiguous().numpy().tobytes()+probe['y'].numpy().tobytes()).hexdigest()
        assert digest==plan['batch_sha256']
    pairs=0; restored_slots=0; max_d=0.; max_q=0.; round_checks=[]
    for batch,item in zip(batches,protocol['schedule']):
        c=batch['round']; assert c==item['cohort']
        snap=torch.load(root/f'raw_round_{c:02d}.pt',map_location='cpu',weights_only=False)
        assert snap['history']=={} and snap['round_index']==c-1
        assert snap['slots']==list(zip(item['clients'],item['profiles']))
        for i,((cid,profile),sent,upload) in enumerate(zip(snap['slots'],snap['dispatched'],snap['uploads'])):
            assert snap['lens'][i]==len(users[str(cid)])
            for key,value in sent.items():
                # Original split_model uses at most the first two dimensions.
                sl=tuple(slice(0,n) for n in value.shape[:2])
                assert torch.equal(value,anchor[key][sl]) and torch.isfinite(upload[key]).all()
            assert state_sha(sent)==batch['dispatched_sha256'][i]
            assert state_sha(upload)==batch['upload_sha256'][i]
            restored_slots+=1
        class Holder:
            region_deltas=ScopeHooks.region_deltas
        h=Holder(); h.slots=snap['slots']
        for region in snap['regions']:
            members=[i for i,(_,p) in enumerate(h.slots) if p in region['profiles']]
            records=[r for r in batch['records'] if r['region']==region['id']]
            if len({h.slots[i][0] for i in members})<2:
                assert not records; continue
            deltas=h.region_deltas(region,snap['uploads'],snap['dispatched'],members)
            refs=references(h,deltas,snap['lens'])
            q,_=semantic_evidence(h.slots,snap['observations'],members,snap['classes'])
            for r in records:
                i=r['sender_slot']; j=r['receiver_slot']; cid=h.slots[i][0]
                assert i in members and j in members and cid!=h.slots[j][0]
                assert cid==r['sender'] and h.slots[j][0]==r['receiver']
                assert r['alpha']==snap['lens'][i]/sum(snap['lens'][k] for k in members)
                cosine=signed_cos(float(torch.dot(deltas[i],refs[i])),float(deltas[i].square().sum()),float(refs[i].square().sum()))
                assert cosine is not None and (cosine<0)==(r['stratum']=='conflict')
                max_d=max(max_d,abs(cosine-r['direction_cosine']))
                assert abs(cosine-r['direction_cosine'])<=1e-12
                assert q.get(cid)==r['q']; max_q=max(max_q,abs(q[cid]-r['q']) if cid in q else 0.)
                if r.get('valid'): assert r['zero_control_max_abs_ce']<=protocol['ce_tolerance']
                pairs+=1
        round_checks.append({'cohort':c,'pairs':len(batch['records']),'mature_anchor_all_slots':True})
    return {'passed':True,'checkpoint_round':400,'restored_slots':restored_slots,'pair_records':pairs,
            'max_direction_replay_error':max_d,'max_q_replay_error':max_q,'round_checks':round_checks,
            'physical_exclusion_native_alpha_original_partition':True,'official_test_loaded':False,
            'global_updates':0,'additional_local_SGD':0,'checkpoint_sha256':sha(checkpoint_path)}


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('root',type=Path)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--check-only',action='store_true',help='Print replay results without creating/replacing evidence')
    p.add_argument('--partition',type=Path,default=ROOT/'data/cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json')
    opt=p.parse_args(); result=verify(opt.root,opt.checkpoint,opt.partition)
    path=opt.root/'offline_verification.json'
    if not opt.check_only:
        with path.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
