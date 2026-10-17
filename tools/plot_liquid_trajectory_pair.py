#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot an auditable two-panel LiquidFL trajectory figure.

The highlighted pair is selected without looking at capacity trajectories:
among personalized clients with the same tier and anchor, choose the pair with
the largest Jensen--Shannon divergence between their pre-training label
histograms.  Panel (a) uses participation events and forward-fills the client's
persistent capacity between controller updates.  Panel (b) shows the same pair
inside the allocation continuum formed by every personalized client.
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from Algorithm.liquid_controller import CapacityProjector
from models.mobileNetV2 import MobileNetV2
from models.resnet import ResNet18_cifar


# Color-blind-safe, print-friendly colors.  Client identity is additionally
# encoded by line style, so the figure remains interpretable in grayscale.
STAGE_COLOR = ('#0072B2', '#D55E00')
TIER_COLOR = {
    'weak': '#0072B2',
    'medium': '#D55E00',
    'strong': '#009E73',
}
PAIR_STYLE = ('-', '--')
INK = '#111111'
MUTED = '#77756f'
GRID = '#dfddd7'
SURFACE = '#ffffff'
OTHER_COLOR = '#858585'


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def _read_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _js_divergence(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left /= max(float(left.sum()), 1.0)
    right /= max(float(right.sum()), 1.0)
    middle = 0.5 * (left + right)

    def kl(first, second):
        mask = first > 0
        return float(np.sum(
            first[mask] * np.log2(first[mask] / second[mask])))

    return 0.5 * kl(left, middle) + 0.5 * kl(right, middle)


def _select_pair(manifest, states, selection_mode, cost_tolerance):
    eligible = []
    profiles = manifest['profile_capacities']
    for key, state in states['clients'].items():
        if not state.get('personalized', False):
            continue
        anchor_profile = state.get('anchor_profile')
        if anchor_profile is None:
            continue
        anchor = np.asarray(
            profiles[int(anchor_profile)], dtype=np.float64)
        if int(np.sum(anchor < 1.0 - 1e-7)) < 2:
            continue
        eligible.append(int(key))

    candidates = []
    for position, left in enumerate(eligible):
        for right in eligible[position + 1:]:
            left_state = states['clients'][str(left)]
            right_state = states['clients'][str(right)]
            if left_state['tier'] != right_state['tier']:
                continue
            if left_state['anchor_profile'] != right_state['anchor_profile']:
                continue
            divergence = _js_divergence(
                manifest['clients'][str(left)]['training_label_hist'],
                manifest['clients'][str(right)]['training_label_hist'])
            left_cost = float(left_state['cost_ratio'])
            right_cost = float(right_state['cost_ratio'])
            relative_cost_gap = abs(left_cost - right_cost) / max(
                left_cost, right_cost, 1e-12)
            left_capacity = np.asarray(
                left_state['capacity'], dtype=np.float64)
            right_capacity = np.asarray(
                right_state['capacity'], dtype=np.float64)
            allocation_distance = float(np.linalg.norm(
                left_capacity[1:] - right_capacity[1:]))
            if selection_mode == 'equal-cost-divergence':
                if relative_cost_gap > float(cost_tolerance):
                    continue
                score = -allocation_distance
            else:
                score = -divergence
            candidates.append((
                score, left, right,
                left_state['tier'], int(left_state['anchor_profile']),
                divergence, relative_cost_gap, allocation_distance))
    if not candidates:
        raise RuntimeError('no protocol-eligible personalized client pair')
    candidates.sort()
    _, left, right, tier, anchor, divergence, cost_gap, distance = (
        candidates[0])
    if selection_mode == 'equal-cost-divergence':
        selection_rule = (
            'maximum final free-stage allocation distance among personalized '
            'clients with the same tier and anchor and a relative final-cost '
            'gap no larger than {:.3%}'.format(float(cost_tolerance)))
    else:
        selection_rule = (
            'maximum pre-training label-distribution JS divergence among '
            'personalized clients with the same tier and anchor')
    return {
        'client_a': left,
        'client_b': right,
        'tier': tier,
        'anchor_profile': anchor,
        'label_js_divergence': divergence,
        'relative_final_cost_gap': cost_gap,
        'final_allocation_distance': distance,
        'selection_rule': selection_rule,
    }


def _persistent_series(events, client_id):
    client_events = sorted(
        (event for event in events
         if int(event['client_id']) == int(client_id)),
        key=lambda event: int(event['participation_event']))
    current = None
    activation_event = None
    rows = []
    triggers = []
    for event in client_events:
        if (activation_event is None and
                (event.get('capacity_before') is not None or
                 event.get('capacity_after') is not None)):
            activation_event = int(event['participation_event'])
        if activation_event is None:
            continue
        relative_event = (
            int(event['participation_event']) - activation_event + 1)
        if current is None and event.get('capacity_before') is not None:
            current = list(event['capacity_before'])
            # Event zero is the shared anchor immediately before the first
            # personalized control event; no pre-activation history is implied.
            rows.append((0, list(current)))
        if event.get('capacity_after') is not None:
            current = list(event['capacity_after'])
        if current is not None:
            rows.append((relative_event, list(current)))
        if event.get('gate_open', False):
            triggers.append(relative_event)
    if not rows:
        raise RuntimeError(
            'selected client {} has no persistent capacity trace'.format(
                client_id))
    deduplicated = {}
    for participation, capacity in rows:
        deduplicated[participation] = capacity
    ordered = sorted(deduplicated.items())
    return ordered, triggers


def _capacity_trajectories(events):
    grouped = collections.defaultdict(list)
    for event in events:
        capacity = event.get('capacity_after')
        if capacity is None:
            continue
        grouped[int(event['client_id'])].append(event)
    for client_id in grouped:
        grouped[client_id].sort(
            key=lambda event: int(event['participation_event']))
    return grouped


def _build_projector(manifest):
    dataset = manifest['dataset']
    num_classes = 100 if dataset == 'cifar100' else 10
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


def _style(axis):
    axis.grid(True, color=GRID, linewidth=0.55, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_color(MUTED)
    axis.spines['bottom'].set_color(MUTED)
    axis.spines['left'].set_linewidth(1.15)
    axis.spines['bottom'].set_linewidth(1.15)
    axis.tick_params(
        labelsize=11.35, colors=INK, direction='out',
        width=1.0, length=4.0, pad=4)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontweight('bold')


def _panel_trajectory(axis, events, pair):
    clients = (pair['client_a'], pair['client_b'])
    series = {}
    triggers = {}
    for client_id in clients:
        series[client_id], triggers[client_id] = _persistent_series(
            events, client_id)

    max_event = max(
        max(row[0] for row in series[client_id])
        for client_id in clients)

    # Give the selected pair context without inventing uncertainty bands.
    # Only clients from the same tier and anchor are shown in the background.
    grouped = collections.defaultdict(list)
    for event in events:
        grouped[int(event['client_id'])].append(event)
    for other_id, other_events in sorted(grouped.items()):
        if other_id in clients:
            continue
        capacity_events = [
            event for event in other_events
            if event.get('capacity_after') is not None]
        if not capacity_events:
            continue
        first = capacity_events[0]
        anchor = next(
            (event.get('anchor_profile') for event in capacity_events
             if event.get('anchor_profile') is not None), None)
        if first.get('tier') != pair['tier']:
            continue
        if anchor is None or int(anchor) != int(pair['anchor_profile']):
            continue
        try:
            other_series, _ = _persistent_series(events, other_id)
        except RuntimeError:
            continue
        other_x = [row[0] for row in other_series]
        for stage_index in (1, 2):
            other_y = [row[1][stage_index] for row in other_series]
            axis.plot(
                other_x, other_y, color=OTHER_COLOR,
                linewidth=0.68, alpha=0.24, zorder=1)

    all_values = []
    for client_index, client_id in enumerate(clients):
        rows = list(series[client_id])
        if rows[-1][0] < max_event:
            rows.append((max_event, list(rows[-1][1])))
        x_values = [row[0] for row in rows]
        for stage_index in (1, 2):
            y_values = [row[1][stage_index] for row in rows]
            all_values.extend(y_values)
            axis.plot(
                x_values, y_values,
                color=STAGE_COLOR[stage_index - 1],
                linestyle=PAIR_STYLE[client_index],
                linewidth=1.60,
                solid_capstyle='round',
                dash_capstyle='butt',
                zorder=3,
                label='Stage {}, Client {}'.format(
                    stage_index + 1, client_id))

    lower = min(all_values)
    upper = max(all_values)
    padding = max(0.025, 0.12 * (upper - lower))
    axis.set_ylim(max(0.0, lower - padding), min(1.02, upper + padding))
    rug_y = axis.get_ylim()[0] + 0.025 * (
        axis.get_ylim()[1] - axis.get_ylim()[0])
    rug_height = 0.022 * (axis.get_ylim()[1] - axis.get_ylim()[0])
    for client_index, client_id in enumerate(clients):
        offset = client_index * rug_height * 1.25
        axis.vlines(
            triggers[client_id], rug_y + offset,
            rug_y + offset + rug_height,
            colors=INK, linewidth=0.68,
            linestyles=PAIR_STYLE[client_index], alpha=0.46)

    axis.set_xlim(-0.5, max_event + 0.8)
    axis.set_xlabel(
        'Participation after activation', fontsize=11.15,
        fontweight='bold', labelpad=6)
    axis.set_ylabel(
        'Stage capacity', fontsize=11.15,
        fontweight='bold', labelpad=7)
    axis.set_title(
        '(a) Client-specific capacity evolution',
        fontsize=11.2, fontweight='bold', pad=9)


def _panel_allocation(axis, trajectories, manifest, projector, pair):
    profiles = manifest['profile_capacities']
    states = _read_json(os.path.join(
        os.path.dirname(os.path.abspath(manifest['_manifest_path'])),
        'latest_states.json'))
    tier_budget = {}
    tier_ceiling = {'weak': 1, 'medium': 3, 'strong': 4}
    for tier, profile_index in tier_ceiling.items():
        tier_budget[tier] = projector.exact_cost(
            profiles[profile_index])

    grid = np.linspace(0.2, 1.0, 80)
    xx, yy = np.meshgrid(grid, grid)
    zz = np.empty_like(xx)
    for row in range(xx.shape[0]):
        for column in range(xx.shape[1]):
            zz[row, column] = projector.exact_cost(
                [1.0, float(xx[row, column]), float(yy[row, column])])
    selected_tier = pair['tier']
    if selected_tier in ('weak', 'medium'):
        level = tier_budget[selected_tier]
        budget_contour = axis.contour(
            xx, yy, zz, levels=[level],
            colors=[MUTED], linewidths=0.95,
            linestyles='dashdot', alpha=0.72)
        manual_position = (
            (0.67, 0.64) if selected_tier == 'weak'
            else (0.72, 0.84))
        axis.clabel(
            budget_contour, levels=[level], inline=True,
            fmt='{}-tier budget'.format(selected_tier.capitalize()),
            fontsize=8.0, colors=[MUTED], manual=[manual_position])

    highlighted = {pair['client_a'], pair['client_b']}
    distinct_signatures = set()
    for client_id, client_events in sorted(trajectories.items()):
        tier = client_events[0]['tier']
        x_values = [event['capacity_after'][1] for event in client_events]
        y_values = [event['capacity_after'][2] for event in client_events]
        for event in client_events:
            signature = event.get('hard_signature_after')
            if signature:
                distinct_signatures.add(tuple(signature))
        if client_id in highlighted:
            continue
        axis.plot(
            x_values, y_values, color=OTHER_COLOR,
            linewidth=0.70, alpha=0.23)
        axis.plot(
            x_values[-1], y_values[-1], 'o',
            color=OTHER_COLOR, markersize=2.6, alpha=0.42)

    for client_index, client_id in enumerate(
            (pair['client_a'], pair['client_b'])):
        client_events = trajectories[client_id]
        x_values = [event['capacity_after'][1] for event in client_events]
        y_values = [event['capacity_after'][2] for event in client_events]
        axis.plot(
            x_values, y_values, color=INK,
            linestyle=PAIR_STYLE[client_index], linewidth=1.95,
            solid_capstyle='round', dash_capstyle='butt',
            label='Client {}'.format(client_id))
        axis.plot(
            x_values[-1], y_values[-1],
            'o' if client_index == 0 else 's',
            color=INK, markerfacecolor=SURFACE,
            markersize=5.2, markeredgewidth=1.25)

    anchor_offsets = {
        0: (6, -2),
        1: (7, -2),
        2: (7, -2),
        3: (7, -2),
        4: (-19, -12),
    }
    for index, capacity in enumerate(profiles):
        axis.plot(
            capacity[1], capacity[2], 'D', color=INK,
            markeredgecolor=SURFACE, markeredgewidth=0.7,
            markersize=4.8, zorder=5)
        axis.annotate(
            'F{}'.format(index), (capacity[1], capacity[2]),
            xytext=anchor_offsets.get(index, (6, -2)),
            textcoords='offset points', fontsize=8.4,
            fontweight='bold', color=INK)

    axis.text(
        0.03, 0.97,
        '5 base configurations  →  {}\nrealized hard subnetworks'.format(
            len(distinct_signatures)),
        transform=axis.transAxes, ha='left', va='top', fontsize=8.25,
        fontweight='semibold', color=INK,
        bbox=dict(
            boxstyle='round,pad=0.22', facecolor=SURFACE,
            edgecolor=GRID, linewidth=0.65, alpha=0.94))
    axis.set_xlim(0.57, 1.04)
    axis.set_ylim(0.18, 1.035)
    axis.set_xlabel(
        'Stage 2 capacity', fontsize=11.15,
        fontweight='bold', labelpad=6)
    axis.set_ylabel(
        'Stage 3 capacity', fontsize=11.15,
        fontweight='bold', labelpad=7)
    axis.set_title(
        '(b) Realized hard-subnetwork space',
        fontsize=11.2, fontweight='bold', pad=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', required=True)
    parser.add_argument('--out', default='liquid_trajectory_pair')
    parser.add_argument(
        '--selection-mode', default='equal-cost-divergence',
        choices=('equal-cost-divergence', 'label-js'))
    parser.add_argument('--cost-tolerance', type=float, default=0.005)
    args = parser.parse_args()

    trace_dir = os.path.abspath(args.trace)
    manifest_path = os.path.join(trace_dir, 'trace_manifest.json')
    manifest = _read_json(manifest_path)
    manifest['_manifest_path'] = manifest_path
    states = _read_json(os.path.join(trace_dir, 'latest_states.json'))
    events = list(_read_jsonl(os.path.join(
        trace_dir, 'client_events.jsonl')))
    trajectories = _capacity_trajectories(events)
    pair = _select_pair(
        manifest, states, args.selection_mode, args.cost_tolerance)
    projector = _build_projector(manifest)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    figure.patch.set_facecolor(SURFACE)
    for axis in axes:
        axis.set_facecolor(SURFACE)
        _style(axis)
    _panel_trajectory(axes[0], events, pair)
    _panel_allocation(
        axes[1], trajectories, manifest, projector, pair)

    client_a, client_b = pair['client_a'], pair['client_b']
    legend_handles = [
        Line2D([0], [0], color=STAGE_COLOR[0], linewidth=1.9,
               label='Stage 2'),
        Line2D([0], [0], color=STAGE_COLOR[1], linewidth=1.9,
               label='Stage 3'),
        Line2D([0], [0], color=INK, linewidth=1.9, linestyle='-',
               label='Client {}'.format(client_a)),
        Line2D([0], [0], color=INK, linewidth=1.9, linestyle='--',
               label='Client {}'.format(client_b)),
        Line2D([0], [0], color=OTHER_COLOR, linewidth=1.15, alpha=0.72,
               label='Other clients'),
        Line2D([0], [0], color=INK, marker='D', linestyle='None',
               markeredgecolor=SURFACE, markersize=5.0,
               label='Base configuration'),
    ]
    figure.legend(
        handles=legend_handles, loc='lower center', ncol=6,
        frameon=False, handlelength=2.45,
        columnspacing=1.02, bbox_to_anchor=(0.5, 0.052),
        prop={'family': 'serif', 'size': 9.55, 'weight': 'bold'})
    figure.tight_layout(rect=(0.0, 0.135, 1.0, 1.0), w_pad=0.15)

    output_base = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(output_base) or '.', exist_ok=True)
    for extension in ('pdf', 'png'):
        path = '{}.{}'.format(output_base, extension)
        figure.savefig(
            path, dpi=300, bbox_inches='tight', facecolor=SURFACE)
        print('wrote {}'.format(path))
    plt.close(figure)
    with open(
            '{}_selection.json'.format(output_base),
            'w', encoding='utf-8') as handle:
        json.dump(pair, handle, ensure_ascii=False, indent=2)
    print(json.dumps(pair, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
