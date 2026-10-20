"""Read-only audit of real SGD or 3 complete original communication rounds.

Each child uses the unchanged 400-round configuration but stops after a fixed
diagnostic budget. This never marks a short run as a completed experiment.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def equal(a,b):
    import numpy as np
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def differences(a,b):
    import torch
    changed=[k for k in a if not torch.equal(a[k],b[k])]
    return dict(equal=not changed,changed_tensors=len(changed),first_changed=changed[0] if changed else None,
        max_abs=max((float((a[k]-b[k]).abs().max()) for k in changed),default=0.),
        difference_l2=sum(float((a[k]-b[k]).double().square().sum()) for k in changed)**.5)


class BudgetComplete(Exception):pass


def child(options,remaining):
    import torch
    import adapterfl_methods as methods
    import tools.run_evidence_candidate as entry
    from models.Update import LocalUpdate_AdaptiveFL
    from utils.reproducibility import execution_policy
    original_loop=entry.AdaptiveFL
    original_train=LocalUpdate_AdaptiveFL.train
    original_record=methods.RunLog.record
    audit=options.audit_output
    audit.mkdir(parents=True,exist_ok=False)
    def loop(*args,**kwargs):
        try:return original_loop(*args,**kwargs)
        except BudgetComplete:pass
    entry.AdaptiveFL=loop
    if options.mode=='sgd':
        slot_reports=[]
        def repeated(local,round,net):
            initial=copy.deepcopy(net)
            start=methods.rng_state()
            captures=[]
            for _ in range(2):
                methods.restore_rng(start)
                model=copy.deepcopy(initial)
                trace={}
                def inputs(module,args):
                    if 'images' not in trace:trace['images']=args[0].detach().cpu().clone()
                handle=model.register_forward_pre_hook(inputs)
                def outputs(module,args,value):
                    if 'logits' not in trace:trace['logits']=value['output'].detach().cpu().clone()
                output_handle=model.register_forward_hook(outputs)
                original_step=torch.optim.SGD.step
                def step(optimizer,*a,**kw):
                    if 'gradients' not in trace:
                        trace['gradients']={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None}
                    return original_step(optimizer,*a,**kw)
                torch.optim.SGD.step=step
                try:weights=original_train(local,round,model)
                finally:torch.optim.SGD.step=original_step;handle.remove();output_handle.remove()
                trace['weights']=methods.cpu_copy(weights);trace['end_rng']=methods.rng_state();captures.append(trace)
            a,b=captures
            report=dict(mode='same actual client, two complete 5-epoch SGD repeats',slot=len(slot_reports),
                policy=execution_policy(),torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                same_first_images=equal(a['images'],b['images']),same_end_rng=equal(a['end_rng'],b['end_rng']),
                first_forward_logits=differences({'logits':a['logits']},{'logits':b['logits']}),
                first_backward_gradients=differences(a['gradients'],b['gradients']),
                final_client_weights=differences(a['weights'],b['weights']),
                partition_examples=len(local.idxs),local_epochs=local.args.local_ep,batch_size=local.args.local_bs)
            slot_reports.append(report)
            (audit/'sgd_repeat.json').write_text(json.dumps(slot_reports,indent=2))
            if len(slot_reports)==options.slot_budget:raise BudgetComplete()
            methods.restore_rng(a['end_rng'])
            return {k:v.to(local.args.device) for k,v in a['weights'].items()}
        LocalUpdate_AdaptiveFL.train=repeated
    else:
        def record(run,round_index,models,evaluations,*a,**kw):
            original_record(run,round_index,models,evaluations,*a,**kw)
            torch.save(dict(full=methods.cpu_copy(models[-1].state_dict()),rng=methods.rng_state(),
                evaluation=evaluations,slots=a[-2]),audit/f'round{round_index+1}.pt')
            if round_index+1==3:
                run.accuracy.close();run.diagnostics.close()
                raise BudgetComplete()
        methods.RunLog.record=record
    try:
        sys.argv=['run_evidence_candidate.py',*remaining]
        entry.main()
        (audit/'budget_complete.json').write_text(json.dumps(dict(mode=options.mode,round_budget=3 if options.mode=='rounds' else 0,
            completed=True,formal_400_round_experiment=False)))
    finally:
        entry.AdaptiveFL=original_loop;LocalUpdate_AdaptiveFL.train=original_train;methods.RunLog.record=original_record


def compare(left,right,output):
    import torch
    rows=[]
    for i in range(1,4):
        a=torch.load(left/f'round{i}.pt',weights_only=False,map_location='cpu')
        b=torch.load(right/f'round{i}.pt',weights_only=False,map_location='cpu')
        rows.append(dict(round=i,weights=differences(a['full'],b['full']),rng_equal=equal(a['rng'],b['rng']),
            slots_equal=equal(a['slots'],b['slots']),evaluation_equal=equal(a['evaluation'],b['evaluation']),
            left_evaluation=a['evaluation'],right_evaluation=b['evaluation']))
    result=dict(passed=all(r['weights']['equal'] and r['rng_equal'] and r['slots_equal'] and r['evaluation_equal'] for r in rows),rounds=rows)
    output.write_text(json.dumps(result,indent=2))
    if not result['passed']:raise RuntimeError('REF/L02 real GPU reproducibility failed; do not start formal experiments')
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=['sgd','rounds','compare'],required=True)
    p.add_argument('--audit-output',type=Path,required=True);p.add_argument('--left',type=Path);p.add_argument('--right',type=Path)
    p.add_argument('--slot-budget',type=int,choices=[2,10],default=10)
    opt,remaining=p.parse_known_args()
    if opt.mode=='compare':compare(opt.left,opt.right,opt.audit_output)
    else:child(opt,remaining)


if __name__=='__main__':main()
