"""Fixed recipient training-probe CE diagnostic. No CE feedback to training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapterfl_methods import isolated_evaluation_rng
from semcons_scope import ScopeHooks, seed_of, slices, semantic_evidence
from tools.probe_semcons_directions import references, signed_cos
import tools.preflight_semcons_activation as preflight

PROTOCOL={'version':1,'rounds':5,'seed':1,'max_classes':16,'samples_per_class':2,'batch_size':50,
    'senders_per_direction_stratum':2,'recipients_per_sender':2,'direction_strata':'signed region LOO cosine <0 versus >=0; zero norm separate',
    'probe_views':'fixed seed per physical client across all rounds; materialized once',
    'injection':'one actual upload-slot delta; alpha=lens[slot]/sum(lens of ALL covering upload slots)',
    'duplicates':'unique physical senders per stratum; hash representative slot; unique recipients; exclude same physical client',
    'ce_tolerance':1e-6,'primary_rounds':[2,3,4,5],'q_definition':'original region physical-client pooled Q; never recomputed from CE',
    'analysis':'Q/D AUC; leave-round-out fixed OLS D versus D+Q, with and without fixed magnitude/own-gain controls',
    'minimum_incremental_sample':{'each_ce_sign_pairs':10,'distinct_senders':10,'rounds':3},
    'raw_snapshot_budget_bytes':6*1024**3,'feedback_to_training':False}


def fixed_plan(dataset,indices,cid):
    groups={}
    for i in sorted(set(map(int,indices))):groups.setdefault(int(dataset.targets[i]),[]).append(i)
    labels=sorted((y for y,v in groups.items() if len(v)>=2),key=lambda y:seed_of('receiver-ce-classes-v1',1,cid,y))[:16]
    selected=[i for y in labels for i in sorted(groups[y],key=lambda i:seed_of('receiver-ce-index-v1',1,cid,i))[:2]]
    return selected,sorted(groups)


def materialize(dataset,indices,cid):
    with isolated_evaluation_rng():
        seed=seed_of('receiver-ce-fixed-view-v1',1,cid)
        random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)
        if torch.cuda.is_initialized():torch.cuda.manual_seed_all(seed)
        rows=[dataset[i] for i in indices]
    if not rows:return None
    x=torch.stack([a for a,b in rows]);y=torch.tensor([int(b) for a,b in rows],dtype=torch.long)
    digest=hashlib.sha256(x.contiguous().numpy().tobytes()+y.numpy().tobytes()).hexdigest()
    return {'x':x,'y':y,'sha256':digest}


@torch.no_grad()
def ce_vector(net,probe):
    device=next(net.parameters()).device;out=[]
    net.eval()
    # Same materialized ordered tensors and boundaries for baseline and injection.
    for start in range(0,len(probe['y']),PROTOCOL['batch_size']):
        x=probe['x'][start:start+PROTOCOL['batch_size']].to(device)
        y=probe['y'][start:start+PROTOCOL['batch_size']].to(device)
        logits=net(x)['output'];loss=F.cross_entropy(logits,y,reduction='none').double().cpu()
        if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite recipient CE')
        out.append(loss)
    return torch.cat(out)


@torch.no_grad()
def inject(net,region,upload,dispatch,alpha):
    state=net.state_dict()
    for cell in region['cells']:
        k=cell['key'];sl=slices(cell)
        for d,(_,upper) in enumerate(cell['bounds']):
            if any(upper>s[k].shape[d] for s in (state,upload,dispatch)):raise ValueError('Unsupported injection cell')
        state[k][sl].add_((upload[k][sl]-dispatch[k][sl]).to(state[k].device),alpha=float(alpha))


def select_pairs(hook,region,members,cosines):
    round_index=hook.round_index;counts={};pairs=[]
    for name,negative in [('conflict',True),('nonconflict',False)]:
        eligible=[i for i in members if cosines.get(i) is not None and (cosines[i]<0)==negative]
        by_client={}
        for i in eligible:
            cid=hook.slots[i][0];by_client.setdefault(cid,[]).append(i)
        chosen=sorted(by_client,key=lambda c:seed_of('ce-donor-v1',1,round_index,region['id'],name,c))[:2]
        counts[name]={'candidate_slots':len(eligible),'candidate_physical_senders':len(by_client),'selected_physical_senders':len(chosen)}
        for cid in chosen:
            sender=min(by_client[cid],key=lambda i:seed_of('ce-donor-slot-v1',1,round_index,region['id'],cid,i))
            recipients={}
            for j in members:
                c=hook.slots[j][0]
                if c!=cid:recipients.setdefault(c,[]).append(j)
            targets=sorted(recipients,key=lambda c:seed_of('ce-recipient-v1',1,round_index,region['id'],cid,c))[:2]
            for c in targets:
                receiver=min(recipients[c],key=lambda j:seed_of('ce-recipient-slot-v1',1,round_index,region['id'],cid,c,j))
                pairs.append((name,sender,receiver))
    return pairs,counts


class ReceiverProbe:
    def __init__(self,dataset,users,root):
        self.root=root;self.templates=None;self.raw_bytes=0;self.plans={};self.probes={};self.classes={}
        for cid,indices in sorted(users.items()):
            cid=int(cid);plan,classes=fixed_plan(dataset,indices,cid)
            self.plans[cid]=plan;self.classes[cid]=classes
            self.probes[cid]=materialize(dataset,plan,cid)
        (root/'probe_plans.json').write_text(json.dumps({str(c):{'indices':p,'training_classes':self.classes[c],
            'batch_sha256':self.probes[c]['sha256'] if self.probes[c] else None} for c,p in self.plans.items()},indent=2))
        torch.save(self.probes,root/'fixed_probes.pt')

    def model(self,profile,state):
        net=copy.deepcopy(self.templates[profile]).eval();net.load_state_dict(state);return net

    @torch.no_grad()
    def measure(self,hook,uploads,lens,dispatched):
        started=time.perf_counter();records=[];inventory=[];base_cache={};own_cache={}
        def baseline(slot):
            if slot not in base_cache:
                cid,profile=hook.slots[slot];probe=self.probes[cid]
                if probe is None:return None
                net=self.model(profile,dispatched[slot]);a=ce_vector(net,probe);b=ce_vector(net,probe)
                base_cache[slot]=(a,float((a-b).abs().max()));del net
            return base_cache[slot]
        for region in hook.regions:
            members=[i for i,(_,p) in enumerate(hook.slots) if p in region['profiles']]
            distinct={hook.slots[i][0] for i in members}
            info={'region':region['id'],'numel':region['numel'],'members':members,'physical_clients':sorted(distinct)}
            inventory.append(info)
            if len(distinct)<2:
                info['fallback']='insufficient_distinct_clients';continue
            deltas=hook.region_deltas(region,uploads,dispatched,members);refs=references(hook,deltas,lens)
            cosine={i:signed_cos(float(torch.dot(deltas[i],refs[i])),float(deltas[i].square().sum()),float(refs[i].square().sum())) for i in members}
            q,sem=semantic_evidence(hook.slots,hook.observations,members,hook.classes)
            pairs,info['strata']=select_pairs(hook,region,members,cosine);mass=sum(lens[i] for i in members)
            info['zero_norm_direction_slots']=sum(cosine[i] is None for i in members)
            for stratum,i,j in pairs:
                sender,sp=hook.slots[i];receiver,rp=hook.slots[j];alpha=lens[i]/mass
                row={'round':hook.round_index+1,'region':region['id'],'region_numel':region['numel'],'stratum':stratum,
                     'sender':sender,'receiver':receiver,'sender_slot':i,'receiver_slot':j,'sender_profile':sp,'receiver_profile':rp,
                     'sender_duplicate_slots':sum(hook.slots[k][0]==sender for k in members),
                     'receiver_duplicate_slots':sum(hook.slots[k][0]==receiver for k in members),
                     'covering_slot_count':len(members),'covering_physical_clients':len(distinct),'native_mass':mass,
                     'sender_weight':lens[i],'alpha':alpha,'direction_cosine':cosine[i],'direction_risk':max(0.,-cosine[i]),
                     'pair_direction_cosine':signed_cos(float(torch.dot(deltas[i],deltas[j])),float(deltas[i].square().sum()),float(deltas[j].square().sum())),
                     'q':q.get(sender),'q_valid_classes':sem['valid_classes'].get(sender,[]),
                     'scaled_delta_l2':alpha*float(deltas[i].norm())}
                b=baseline(j);s=baseline(i)
                if b is None or s is None:
                    row.update(valid=False,reason='no_fixed_probe');records.append(row);continue
                if i not in own_cache:
                    net=self.model(sp,uploads[i]);own_cache[i]=ce_vector(net,self.probes[sender]);del net
                net=self.model(rp,dispatched[j]);inject(net,region,uploads[i],dispatched[i],alpha)
                after=ce_vector(net,self.probes[receiver]);del net
                difference=after-b[0];tol=PROTOCOL['ce_tolerance'];zero=max(b[1],s[1])
                classes=[];labels=self.probes[receiver]['y']
                for y in sorted(set(labels.tolist())):
                    mask=labels==y;delta=float(difference[mask].mean())
                    classes.append({'class':y,'n':int(mask.sum()),'ce_before':float(b[0][mask].mean()),'ce_after':float(after[mask].mean()),
                                    'delta_ce':delta,'sender_has_training_class':y in self.classes[sender],
                                    'in_sender_q_classes':y in row['q_valid_classes']})
                delta=float(difference.mean())
                row.update(valid=zero<=tol,reason=None if zero<=tol else 'zero_control_exceeds_fixed_tolerance',
                    ce_before=float(b[0].mean()),ce_after=float(after.mean()),delta_ce=delta,
                    outcome='worse' if delta>tol else 'better' if delta< -tol else 'neutral',zero_control_max_abs_ce=zero,
                    sender_own_ce_before=float(s[0].mean()),sender_own_ce_after=float(own_cache[i].mean()),
                    sender_own_gain=float((s[0]-own_cache[i]).mean()),receiver_probe_sha256=self.probes[receiver]['sha256'],
                    receiver_probe_examples=len(labels),classes=classes)
                records.append(row)
        # Retain the real inputs for future offline recomputation; never put these in Git.
        snapshot={'slots':hook.slots,'lens':lens,'regions':hook.regions,'observations':hook.observations,
                  'round_index':hook.round_index,'classes':hook.classes,'history':dict(hook.history),
                  'uploads':[{k:v.detach().cpu().clone() for k,v in s.items()} for s in uploads],
                  'dispatched':[{k:v.detach().cpu().clone() for k,v in s.items()} for s in dispatched]}
        size=sum(v.numel()*v.element_size() for key in ('uploads','dispatched') for s in snapshot[key] for v in s.values())
        self.raw_bytes+=size
        if self.raw_bytes>PROTOCOL['raw_snapshot_budget_bytes']:raise RuntimeError('Fixed raw snapshot budget exceeded; no retry')
        torch.save(snapshot,self.root/f'raw_round_{hook.round_index+1:02d}.pt')
        return {'round':hook.round_index+1,'records':records,'inventory':inventory,'raw_tensor_bytes':size,
                'diagnostic_seconds':time.perf_counter()-started,'baseline_slots':len(base_cache),'sender_full_post_forwards':len(own_cache)}


def observed(original,observer):
    def aggregate(hook,uploads,lens,previous,dispatched=None):
        result=None
        if hook.method=='s07':
            with isolated_evaluation_rng():result=observer.measure(hook,uploads,lens,dispatched)
        actual=original(hook,uploads,lens,previous,dispatched)
        if result is not None:
            result['actual_gate_fraction']=hook.round_diagnostics['gate_trigger_region_slot_fraction']
            with (observer.root/'receiver_ce.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        return actual
    return aggregate


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output-root',type=Path,required=True);opt=p.parse_args()
    root=opt.output_root;root.mkdir(parents=True,exist_ok=False)
    (root/'protocol.json').write_text(json.dumps(dict(PROTOCOL,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    old_loader=preflight.get_dataset;old_init=ScopeHooks.__init__;old_aggregate=ScopeHooks.aggregate
    holder={}
    def loaded(args):
        train,test,users=old_loader(args)
        with isolated_evaluation_rng():holder['observer']=ReceiverProbe(train,users,root)
        return train,test,users
    def initialize(hook,args,models):
        old_init(hook,args,models);holder['observer'].templates=models
    def aggregate(hook,*args,**kwargs):return observed(old_aggregate,holder['observer'])(hook,*args,**kwargs)
    saved=sys.argv
    try:
        preflight.get_dataset=loaded;ScopeHooks.__init__=initialize;ScopeHooks.aggregate=aggregate
        sys.argv=['preflight_semcons_activation.py','--mode','s07','--rounds','5','--gpu','0','--output-root',str(root)]
        preflight.main()
    finally:
        preflight.get_dataset=old_loader;ScopeHooks.__init__=old_init;ScopeHooks.aggregate=old_aggregate;sys.argv=saved
    (root/'diagnostic_completed.json').write_text(json.dumps({'rounds':5,'formal400':False,'success':True}))


if __name__=='__main__':main()
