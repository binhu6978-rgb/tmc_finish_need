"""Recompute a saved recipient probe on CPU, with no SGD and no dataset access."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapterfl_methods import rng_state
from models.Fed import get_model_list
from semcons_scope import ScopeHooks
from tools.receiver_ce_probe import ReceiverProbe,observed


def tensor_digest(states):
    digest=hashlib.sha256()
    for state in states:
        for k,v in state.items():digest.update(k.encode());digest.update(v.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_digest():
    state=rng_state();h=hashlib.sha256(repr(state['python']).encode())
    h.update(state['numpy'][1].tobytes());h.update(repr(state['numpy'][2:]).encode())
    h.update(state['torch'].numpy().tobytes())
    for v in state['cuda']:h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);p.add_argument('--round',type=int,default=2);p.add_argument('--output',type=Path,required=True);opt=p.parse_args()
    torch.set_num_threads(2);opt.output.mkdir(parents=True,exist_ok=False)
    raw=torch.load(opt.root/f'raw_round_{opt.round:02d}.pt',map_location='cpu',weights_only=False)
    args=SimpleNamespace(method='s07',conflict_lambda=4.,num_classes=raw['classes'],num_channels=3,seed=1,local_bs=50,
        model='resnet',width_ration=[.4,.66,1.],depth_saved=[2,3,4],device=torch.device('cpu'))
    with contextlib.redirect_stdout(io.StringIO()):models,_=get_model_list(args)
    hook=ScopeHooks(args,models)
    if hook.regions!=raw['regions']:raise ValueError('Saved geometry differs')
    full=[i for i,(_,profile) in enumerate(raw['slots']) if profile==6]
    if not full:raise ValueError('Replay verification requires an actual Full dispatch snapshot')
    hook.round_index=raw['round_index'];hook.slots=raw['slots'];hook.observations=raw['observations'];hook.history=raw['history']
    observer=ReceiverProbe.__new__(ReceiverProbe);observer.root=opt.output;observer.templates=models;observer.raw_bytes=0
    observer.probes=torch.load(opt.root/'fixed_probes.pt',map_location='cpu',weights_only=False)
    plans=json.loads((opt.root/'probe_plans.json').read_text());observer.classes={int(c):v['training_classes'] for c,v in plans.items()}
    previous=raw['dispatched'][full[0]];reference=copy.deepcopy(hook)
    before_inputs=tensor_digest(raw['uploads']+raw['dispatched']);before_models=tensor_digest([m.state_dict() for m in models])
    expected=reference.aggregate(raw['uploads'],raw['lens'],previous,raw['dispatched'])
    before_rng=rng_digest()
    actual=observed(ScopeHooks.aggregate,observer)(hook,raw['uploads'],raw['lens'],previous,raw['dispatched'])
    result={'round':opt.round,'no_SGD':True,'no_dataset_access':True,
        'native_hook_output_exactly_equal':all(torch.equal(actual[k],expected[k]) for k in actual),
        'history_equal':hook.history==reference.history,'rng_equal':before_rng==rng_digest(),
        'uploads_and_dispatch_unchanged':before_inputs==tensor_digest(raw['uploads']+raw['dispatched']),
        'template_models_unchanged':before_models==tensor_digest([m.state_dict() for m in models]),
        'CE_backend':'CPU; numerical values not pooled with original GPU observations'}
    if not all(result[k] for k in ('native_hook_output_exactly_equal','history_equal','rng_equal','uploads_and_dispatch_unchanged','template_models_unchanged')):
        raise AssertionError(result)
    (opt.output/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
