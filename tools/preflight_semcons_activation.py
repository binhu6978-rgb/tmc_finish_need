"""Bounded full-partition activation diagnostic; calls the unchanged native loop.

This is not the 400-round queue. Never changes gate rules or local training.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapterfl_methods import isolated_evaluation_rng, prepare_output
from semcons_scope import ScopeHooks
from tools.smoke_semcons_scope10 import AdaptiveFL, args_parser, get_dataset, _build_record_path, set_random_seed, ResNet18_cifar

MODES = ('s03', 's04', 's06', 's07', 's08', 's09')


def observed_aggregate(original, path):
    """Diagnostics on real upload buffers. Counterfactual output is never trained."""
    def aggregate(hook, uploads, lens, previous, dispatched=None):
        begin = time.perf_counter()
        # Clone before the real call increments history. Model tensors are not in hook.
        alternate = copy.deepcopy(hook) if hook.method == 's07' else None
        out = original(hook, uploads, lens, previous, dispatched)
        extra = {'round': hook.round_index + 1, 'mode': hook.method}
        with isolated_evaluation_rng():
            if hook.method == 's04':
                # S04 does not use direction in its decision; observe it afterwards.
                values = {}
                for region, record in zip(hook.regions, hook.round_diagnostics['regions']):
                    if record['client_count'] >= 2:
                        deltas = hook.region_deltas(region, uploads, dispatched, record['slots'])
                        values[str(region['id'])] = {str(i): v for i, v in hook.direction(deltas, lens).items()}
                extra['direction_observation_only'] = values
            if alternate is not None:
                alternate.method = 's09'
                alternative = original(alternate, uploads, lens, previous, dispatched)
                square = 0.; changed = 0
                for key in out:
                    delta = (out[key] - alternative[key]).double()
                    square += float(delta.square().sum())
                    changed += int(delta.ne(0).sum())
                a = hook.round_diagnostics['regions']; b = alternate.round_diagnostics['regions']
                extra['same_input_s07_s09'] = {
                    'gates_identical': [r['gates'] for r in a] == [r['gates'] for r in b],
                    'history_identical': hook.history == alternate.history,
                    'aggregate_difference_l2': square ** .5,
                    'aggregate_changed_coordinates': changed,
                    's07_actual_changed_fraction': hook.round_diagnostics['actual_changed_parameter_fraction_full'],
                    's09_actual_changed_fraction': alternate.round_diagnostics['actual_changed_parameter_fraction_full'],
                    's07_learning_l2': hook.round_diagnostics['comparable_applied_learning_update_l2'],
                    's09_learning_l2': alternate.round_diagnostics['comparable_applied_learning_update_l2'],
                    'regions': [{'region_id': x['region_id'], 'mass_ratio': x.get('gated_mass_ratio'),
                                 's07_l2': x.get('applied_learning_update_l2'),
                                 's09_l2': y.get('applied_learning_update_l2')} for x, y in zip(a, b)]}
                del alternative
        extra['aggregate_including_diagnostic_seconds'] = time.perf_counter() - begin
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(extra) + '\n')
        return out
    return aggregate


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--rounds', type=int, choices=range(3, 11), default=10)
    parser.add_argument('--gpu', type=int, choices=(-1, 0), default=0)
    parser.add_argument('--output-root', required=True)
    opt = parser.parse_args()
    saved = sys.argv
    try:
        sys.argv = ['main_fed_ori.py', '--method', opt.mode, '--epochs', str(opt.rounds),
                    '--seed', '1', '--gpu', str(opt.gpu), '--generate_data', '0', '--data_beta', '.3',
                    '--eval-profiles', 'full', '--conflict-lambda', '4',
                    '--method-run-name', 'activation_' + str(opt.rounds), '--method-output-root', opt.output_root]
        args = args_parser()
    finally:
        sys.argv = saved
    torch.set_num_threads(2)
    args.model = 'resnet'; args.dataset = 'cifar100'; args.algorithm = 'AdaptiveFL'
    args.device = torch.device('cpu' if opt.gpu == -1 else 'cuda:0')
    set_random_seed(1)
    record = Path(_build_record_path(args))
    if not record.is_file():
        raise FileNotFoundError('Original saved partition required; no partition generation allowed')
    train, test, users = get_dataset(args)
    args._method_partition_sha256 = hashlib.sha256(record.read_bytes()).hexdigest()
    args.scope_activation_provenance = {
        'kind': 'full-data short mechanism preflight, not a formal performance experiment',
        'full_data': True, 'train_examples': len(train), 'test_examples': len(test),
        'clients': len(users), 'partition_memberships': sum(len(v) for v in users.values()),
        'unique_training_indices': len(set(int(i) for v in users.values() for i in v)),
        'examples_per_client': {str(k): len(v) for k, v in users.items()},
        'partition_sha256': args._method_partition_sha256,
        'argv': saved, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'same_input_replay': 'S07 buffers/history replayed through S09; result discarded; no training feedback'}
    path = prepare_output(args)
    (path / 'preflight_provenance.json').write_text(json.dumps(args.scope_activation_provenance, indent=2))
    original = ScopeHooks.aggregate
    ScopeHooks.aggregate = observed_aggregate(original, path / 'activation_observations.jsonl')
    resource = {'pid': os.getpid(), 'device': str(args.device), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    if opt.gpu == 0:
        torch.cuda.reset_peak_memory_stats()
        resource['gpu_name'] = torch.cuda.get_device_name()
    started = time.perf_counter()
    try:
        # Reproduce the original entry's unused initial Full construction for RNG parity.
        unused = ResNet18_cifar(num_channels=args.num_channels, num_classes=args.num_classes)
        AdaptiveFL(args, train, test, users)
        resource['status'] = 'completed_short_preflight'
    except BaseException:
        resource['status'] = 'failed'
        raise
    finally:
        ScopeHooks.aggregate = original
        resource['wall_seconds'] = time.perf_counter() - started
        if opt.gpu == 0:
            resource['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            resource['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        (path / 'preflight_resources.json').write_text(json.dumps(resource, indent=2))


if __name__ == '__main__':
    main()
