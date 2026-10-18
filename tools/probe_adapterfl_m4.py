"""Paired gradient-scale code probe, NOT a federated performance experiment.

Use: python -m tools.probe_adapterfl_m4 --output-dir runs/m4_probe --device cpu
Original seven ResNets, Full local SGD, same initial weights/batches/seed.
Small synthetic tensors are explicitly labelled; lambda=1 is never tuned here.
"""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys

import torch

from adapterfl_methods import MethodHooks,cpu_copy,rng_state,restore_rng
from models.Fed import get_model_list
from models.Update import LocalUpdate_AdaptiveFL
from utils.options import args_parser
from utils.set_seed import set_random_seed


def probe(output,device='cpu',steps=3):
    if steps<2 or steps>5:
        raise ValueError('This lightweight code probe supports 2--5 local steps only')
    path=Path(output);path.mkdir(parents=True,exist_ok=False)
    argv=sys.argv
    try:
        sys.argv=[argv[0]];args=args_parser()
    finally:
        sys.argv=argv
    args.device=torch.device(device);args.gpu=0 if args.device.type=='cuda' else -1
    args.method='m4';args.model='resnet';args.dataset='cifar100';args.num_classes=100
    args.local_bs=2;args.local_ep=1
    set_random_seed(1)
    with contextlib.redirect_stdout(io.StringIO()):
        models,profiles=get_model_list(args)
    hooks=MethodHooks(args,models)
    initial=cpu_copy(models[-1].state_dict())
    data=torch.utils.data.TensorDataset(torch.randn(steps*2,3,32,32),torch.arange(steps*2)%100)
    before=rng_state()
    plain=LocalUpdate_AdaptiveFL(args,data,range(len(data)))
    baseline=cpu_copy(plain.train(0,copy.deepcopy(models[-1])))
    restore_rng(before)
    control=LocalUpdate_AdaptiveFL(args,data,range(len(data)))
    # Numerical/instrumentation floor: same diagnostic graph but exactly zero
    # proximal gradient. This is a code probe, not another experiment candidate.
    control.adapterfl_prox=(hooks.shared_shapes,0.)
    control.adapterfl_prox_diagnostic_steps=steps
    zero_control=cpu_copy(control.train(0,copy.deepcopy(models[-1])))
    restore_rng(before)
    local=LocalUpdate_AdaptiveFL(args,data,range(len(data)))
    hooks.configure_local(local);local.adapterfl_prox_diagnostic_steps=steps
    constrained=cpu_copy(local.train(0,copy.deepcopy(models[-1])))
    difference,baseline_update,changed,control_difference,m4_control_difference=0.,0.,0,0.,0.
    for k,v in baseline.items():
        d=constrained[k].double()-v.double()
        difference+=float(d.square().sum())
        baseline_update+=float((v.double()-initial[k].double()).square().sum())
        changed+=int(torch.count_nonzero(d))
        control_difference+=float((zero_control[k].double()-v.double()).square().sum())
        m4_control_difference+=float((constrained[k].double()-zero_control[k].double()).square().sum())
    result={'kind':'paired_synthetic_local_code_probe_not_performance',
        'device':str(args.device),'torch':torch.__version__,'seed':1,'local_steps':steps,'batch_size':2,
        'image_shape':[3,32,32],'profiles':profiles,'lambda_unchanged':args.shared_prox_lambda,
        'metrics':local.adapterfl_metrics,'parameter_difference_from_baseline_l2':difference**.5,
        'baseline_update_l2':baseline_update**.5,
        'difference_relative_to_baseline_update':(difference/baseline_update)**.5 if baseline_update else None,
        'changed_coordinates':changed,'same_initial_weights_and_training_rng':True,
        'zero_prox_diagnostic_control_vs_baseline_l2':control_difference**.5,
        'm4_vs_zero_prox_diagnostic_control_l2':m4_control_difference**.5,
        'numerical_control_note':'Differences include float32/backward/instrumentation variation; do not attribute the whole parameter difference to the proximal force',
        'dimension_based_proposal':'For a desired unnormalized curvature mu, lambda_mean=N_shared*mu. Alternatively measure rho*task_shared_gradient_l2/proximal_gradient_l2 at a prespecified training-only probe. Neither is applied here.'}
    (path/'probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    torch.save({'baseline':baseline,'m4':constrained,'zero_prox_diagnostic_control':zero_control},path/'paired_final.pt')
    print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','metrics')},indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    torch.set_num_threads(2)
    probe(args.output_dir,args.device)


if __name__=='__main__':
    main()
