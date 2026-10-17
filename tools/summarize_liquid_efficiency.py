#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize LiquidFL compute and communication overhead from a trace run."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Algorithm.liquid_controller import CapacityProjector
from models.mobileNetV2 import MobileNetV2
from models.resnet import ResNet18_cifar


def _read_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _projector(manifest):
    num_classes = 100 if manifest['dataset'] == 'cifar100' else 10
    if manifest['model'] == 'resnet':
        model = ResNet18_cifar(
            num_channels=3, num_classes=num_classes,
            track_running_stats=False, stage_scales=[1.0, 1.0, 1.0])
    else:
        model = MobileNetV2(
            channels=3, num_classes=num_classes, trs=False,
            stage_scales=[1.0, 1.0, 1.0])
    return CapacityProjector(
        model, manifest['model'], min_capacity=0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', default='')
    parser.add_argument(
        '--training-stat-scalars', type=int, default=4,
        help='per selected client: loss, loss drop, grad norm, loss variance')
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    trace_dir = os.path.join(run_dir, 'trace')
    manifest = _read_json(os.path.join(trace_dir, 'trace_manifest.json'))
    events = list(_read_jsonl(os.path.join(
        trace_dir, 'client_events.jsonl')))
    rounds = list(_read_jsonl(os.path.join(
        trace_dir, 'server_rounds.jsonl')))
    projector = _projector(manifest)

    run_args = manifest['arguments']
    coefficient = float(run_args.get('liquid_macs_coeff', 3.0))
    feedback_bs = int(run_args.get('liquid_feedback_bs', 32))
    full_forward_macs = float(projector.estimate_forward_macs([1, 1, 1]))
    sensitivity_per_probe = (
        coefficient * feedback_bs * full_forward_macs)

    event_epsilon = []
    probe_forward_total = 0.0
    sensitivity_total = 0.0
    sensitivity_probes_total = 0
    control_events_total = 0
    q_clients_by_round = {}
    for event in events:
        if not event.get('control_event', False):
            continue
        control_events_total += 1
        capacities = event.get('task_capacities') or []
        profiles = event.get('dispatched_profiles') or []
        processed = float(
            event.get('local_metrics', {}).get('processed_samples', 0.0))
        hard = coefficient * processed * sum(
            float(projector.estimate_forward_macs(capacity))
            for capacity in capacities)

        control_profile = event.get('control_profile')
        probe_forward = 0.0
        if control_profile is not None:
            probe_capacity = manifest['profile_capacities'][
                int(control_profile)]
            probe_forward = feedback_bs * float(
                projector.estimate_forward_macs(probe_capacity))
        probe_forward_total += probe_forward

        controlled = [
            capacity for capacity, profile in zip(capacities, profiles)
            if int(profile) != 4]
        probes = (
            len(controlled) if event.get('q_feedback') is not None else 0)
        sensitivity = probes * sensitivity_per_probe
        feedback = probe_forward + sensitivity
        if hard > 0.0:
            event_epsilon.append(feedback / hard)
        sensitivity_total += sensitivity
        sensitivity_probes_total += probes
        if event.get('q_feedback') is not None:
            round_idx = int(event['round'])
            q_clients_by_round[round_idx] = (
                q_clients_by_round.get(round_idx, 0) + 1)

    feedback_total = probe_forward_total + sensitivity_total

    final_summary = rounds[-1]['round_summary']
    total_macs = float(final_summary['cumulative_macs'])
    kd_macs = float(final_summary['cumulative_kd_macs'])
    hard_macs = total_macs - kd_macs

    selected_unique_total = int(sum(
        int(row['selected_unique_clients']) for row in rounds))
    q_client_events = int(sum(q_clients_by_round.values()))
    float_bytes = 4
    stats_total_bytes = (
        selected_unique_total * int(args.training_stat_scalars) * float_bytes)
    probe_loss_total_bytes = control_events_total * float_bytes
    q_total_bytes = q_client_events * 3 * float_bytes
    metadata_per_round = []
    control_by_round = {}
    for event in events:
        if event.get('control_event', False):
            round_idx = int(event['round'])
            control_by_round[round_idx] = control_by_round.get(round_idx, 0) + 1
    for row in rounds:
        round_idx = int(row['round'])
        metadata_per_round.append(
            int(row['selected_unique_clients'])
            * int(args.training_stat_scalars) * float_bytes
            + control_by_round.get(round_idx, 0) * float_bytes
            + q_clients_by_round.get(round_idx, 0) * 3 * float_bytes)

    dispatched_parameters = float(sum(
        int(row['dispatched_parameter_sum']) for row in rounds))
    actual_parameters = float(sum(
        int(row['actual_parameter_sum']) for row in rounds))
    round_payload_ratios = np.asarray([
        float(row['model_payload_ratio_vs_dispatch']) for row in rounds],
        dtype=np.float64)
    epsilon = np.asarray(event_epsilon, dtype=np.float64)
    result = {
        'rounds': len(rounds),
        'hard_subnetwork_training_macs': hard_macs,
        'kd_macs': kd_macs,
        'feedback_macs': feedback_total,
        'feedback_probe_forward_macs': probe_forward_total,
        'feedback_sensitivity_macs': sensitivity_total,
        'control_probe_events': control_events_total,
        'capacity_sensitivity_probes': sensitivity_probes_total,
        'feedback_amortized_over_hard_training': (
            feedback_total / hard_macs),
        'feedback_amortized_over_training_including_kd': (
            feedback_total / total_macs),
        'epsilon_fb_client_event': {
            'definition': 'feedback_macs / controlled-task hard-training_macs',
            'count': int(epsilon.size),
            'mean': float(epsilon.mean()),
            'p95': float(np.quantile(epsilon, 0.95)),
            'max': float(epsilon.max()),
        },
        'feedback_uplink': {
            'capacity_sensitivity_bytes_per_triggered_client': 12,
            'capacity_sensitivity_total_bytes': q_total_bytes,
            'training_statistics_total_bytes': stats_total_bytes,
            'probe_loss_total_bytes': probe_loss_total_bytes,
            'combined_mean_bytes_per_round': float(np.mean(metadata_per_round)),
            'combined_peak_bytes_per_round': int(max(metadata_per_round)),
        },
        'model_payload': {
            'aggregate_ratio_vs_dispatch': (
                actual_parameters / dispatched_parameters),
            'mean_round_ratio_vs_dispatch': float(
                round_payload_ratios.mean()),
            'peak_round_ratio_vs_dispatch': float(
                round_payload_ratios.max()),
        },
        'assumptions': {
            'float_format': 'float32',
            'feedback_batch_size': feedback_bs,
            'forward_backward_multiplier': coefficient,
            'training_stat_scalars': int(args.training_stat_scalars),
            'feedback_proxy_capacity': [1.0, 1.0, 1.0],
        },
    }
    output = (
        os.path.abspath(args.output) if args.output
        else os.path.join(run_dir, 'efficiency_summary.json'))
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('wrote {}'.format(output))


if __name__ == '__main__':
    main()
