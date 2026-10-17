#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the two LiquidFL "liquid behaviour" subfigures from a trace run.

Panel (a): per-client realized cost ratio across communication rounds, showing
           that personalized clients are promoted at different rounds and then
           evolve at client-specific rates toward (or short of) their own tier
           budget.
Panel (b): the (layer3, layer4) allocation plane. Because every profile in the
           nested pool fixes layer2 = 1.0, this plane is the complete free
           allocation space, so client trajectories and the tier budget
           iso-cost frontiers can be drawn without projection loss.

Usage:
    python tools/plot_liquid_figures.py \
        --trace result_liquid_visual_trace/cifar100_resnet_beta0.3_seed42_.../trace \
        --out figures/liquid_dynamics
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
import numpy as np
from matplotlib.lines import Line2D

# Validated categorical slots (light surface, all-pairs CVD-safe).
TIER_COLOR = {'weak': '#2a78d6', 'medium': '#eb6834', 'strong': '#1baf7a'}
TIER_LABEL = {'weak': 'Weak', 'medium': 'Medium', 'strong': 'Strong'}
TIER_ORDER = ('weak', 'medium', 'strong')

INK = '#0b0b0b'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

# Typography: axis frame is deliberately heavier than the annotation layer so
# the figure stays readable when scaled down to a single AAAI column.
FS_TITLE = 9.0
FS_LABEL = 9.5
FS_TICK = 8.5
FS_ANNOT = 7.0
FS_LEGEND = 8.0

# Each tier's budget equals the cost of its ceiling profile in the nested pool
# (see LiquidStateManager: weak -> F1, medium -> F3, strong -> F4). Stated in
# the figure so the five anchors and the three tiers can be cross-referenced.
TIER_CEILING_PROFILE = {'weak': 1, 'medium': 3, 'strong': 4}

# Representative clients for panel (a): per tier, one fast riser and one that
# plateaus below its budget, so rate and endpoint heterogeneity are both visible.
PANEL_A_CLIENTS = [
    ('weak', 35, '-'), ('weak', 12, '--'),
    ('medium', 53, '-'), ('medium', 40, '--'),
    ('strong', 90, '-'), ('strong', 77, '--'),
]


def load_trace(trace_dir):
    events = [json.loads(line) for line in
              open(os.path.join(trace_dir, 'client_events.jsonl'), encoding='utf-8')]
    manifest = json.load(
        open(os.path.join(trace_dir, 'trace_manifest.json'), encoding='utf-8'))
    traj = collections.defaultdict(list)
    for event in events:
        if event.get('capacity_after'):
            traj[event['client_id']].append(event)
    for client_id in traj:
        traj[client_id].sort(key=lambda e: e['round'])
    return traj, manifest


def build_cost_fn(args_from_manifest, model_name, num_classes):
    """Return cost(capacity)->parameter ratio using the real projector."""
    import torch  # noqa: F401  (imported lazily; heavy)
    from Algorithm.liquid_controller import CapacityProjector
    from models.resnet import ResNet18_cifar
    from models.mobileNetV2 import MobileNetV2
    if model_name == 'resnet':
        model = ResNet18_cifar(num_channels=3, num_classes=num_classes,
                               track_running_stats=False, stage_scales=[1.0, 1.0, 1.0])
    else:
        model = MobileNetV2(channels=3, num_classes=num_classes, trs=False,
                            stage_scales=[1.0, 1.0, 1.0])
    projector = CapacityProjector(model, model_name, min_capacity=0.2)
    return projector.exact_cost


def panel_a(ax, traj, tier_budget):
    # Every personalized client is drawn faintly, so the highlighted few are
    # visibly representative of the population rather than hand-picked.
    for client_id, events in sorted(traj.items()):
        tier = events[0]['tier']
        ax.plot([e['round'] for e in events],
                [e['cost_ratio_after'] for e in events],
                '-', color=TIER_COLOR[tier], lw=0.7, alpha=0.22, zorder=2)

    for tier in TIER_ORDER:
        budget = tier_budget[tier]
        ax.axhline(budget, color=TIER_COLOR[tier], lw=1.0, ls=(0, (4, 2)),
                   alpha=0.85, zorder=3)
        ax.text(986, budget + 0.018, f'{TIER_LABEL[tier]} budget',
                fontsize=FS_ANNOT, color=TIER_COLOR[tier], va='bottom',
                ha='right', zorder=6, fontweight='bold')

    for tier, client_id, style in PANEL_A_CLIENTS:
        if client_id not in traj:
            continue
        events = traj[client_id]
        rounds = [e['round'] for e in events]
        cost = [e['cost_ratio_after'] for e in events]
        ax.plot(rounds, cost, style, color=TIER_COLOR[tier], lw=1.7,
                solid_capstyle='round', dash_capstyle='round', zorder=5)
        ax.plot(rounds[0], cost[0], 'o', color=TIER_COLOR[tier], ms=4.2,
                mec='#fcfcfb', mew=1.0, zorder=6)

    # Name the phenomena a reviewer should take away. (The trigger marker
    # itself is explained in the shared legend, so it needs no callout here.)
    ax.annotate('plateaus below its own budget',
                xy=(690, 0.388), xytext=(455, 0.198),
                fontsize=FS_ANNOT, color=INK, ha='left',
                arrowprops=dict(arrowstyle='->', lw=0.8, color=MUTED,
                                shrinkA=0, shrinkB=3))
    ax.annotate('same tier, different rate',
                xy=(300, 0.667), xytext=(352, 0.865),
                fontsize=FS_ANNOT, color=INK, ha='left',
                arrowprops=dict(arrowstyle='->', lw=0.8, color=MUTED,
                                shrinkA=0, shrinkB=3))

    ax.set_xlim(0, 1000)
    ax.set_ylim(0.15, 1.09)
    ax.set_xlabel('Communication round', fontsize=FS_LABEL, color=INK,
                  fontweight='bold')
    # The exact definition Cost(A)/Cost(1) is given in the caption, keeping the
    # axis label short enough not to crowd the title.
    ax.set_ylabel('Realized cost ratio', fontsize=FS_LABEL, color=INK,
                  fontweight='bold')
    ax.set_title('(a) Capacity evolves per client, within budget',
                 fontsize=FS_TITLE, color=INK, pad=6, fontweight='bold')


def panel_b(ax, traj, cost_fn, profiles, tier_budget):
    # iso-cost frontiers over the complete free allocation plane
    grid = np.linspace(0.18, 1.0, 90)
    xx, yy = np.meshgrid(grid, grid)
    zz = np.empty_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = cost_fn([1.0, float(xx[i, j]), float(yy[i, j])])

    # Weak/medium budget frontiers are proper curves in this plane. The strong
    # budget is the full model, i.e. the single point F4, so it is annotated at
    # that anchor instead of drawn as a (degenerate) contour.
    frontier_label_at = {'weak': (0.735, 0.545), 'medium': (0.775, 0.80)}
    for tier in ('weak', 'medium'):
        level = tier_budget[tier]
        single = ax.contour(xx, yy, zz, levels=[level],
                            colors=[TIER_COLOR[tier]], linewidths=1.0,
                            linestyles='dashed', alpha=0.9, zorder=1)
        labels = ax.clabel(
            single,
            fmt={level: f'{TIER_LABEL[tier]} budget'},
            fontsize=FS_ANNOT, inline=True, inline_spacing=3,
            manual=[frontier_label_at[tier]])
        for text in labels:
            text.set_fontweight('bold')

    # client trajectories through the allocation plane
    signatures = set()
    for client_id, events in sorted(traj.items()):
        tier = events[0]['tier']
        xs = [e['capacity_after'][1] for e in events]
        ys = [e['capacity_after'][2] for e in events]
        for event in events:
            signatures.add(tuple(event.get('hard_signature_after') or []))
        ax.plot(xs, ys, '-', color=TIER_COLOR[tier], lw=0.9, alpha=0.45,
                zorder=2)
        ax.plot(xs[-1], ys[-1], 'o', color=TIER_COLOR[tier], ms=3.6,
                mec='#fcfcfb', mew=0.7, alpha=0.95, zorder=4)

    # fixed profile anchors of the nested pool
    offsets = {0: (8, -3), 1: (8, -9), 2: (8, 2), 3: (8, -4), 4: (8, -4)}
    for idx, capacity in enumerate(profiles):
        ax.plot(capacity[1], capacity[2], 'D', color=INK, ms=4.6,
                mec='#fcfcfb', mew=0.8, zorder=5)
        ax.annotate(f'F{idx}', (capacity[1], capacity[2]),
                    textcoords='offset points', xytext=offsets.get(idx, (8, -3)),
                    fontsize=FS_ANNOT, color=INK, zorder=5, ha='left',
                    fontweight='bold')

    # Precise wording: these are the distinct hard sub-networks actually
    # instantiated and trained over the run (transient states included), not
    # the number of distinct endpoints -- the caption states both. The second
    # line cross-references the five anchors against the three resource tiers.
    ceiling = ',  '.join(
        f'{TIER_LABEL[t]}=F{TIER_CEILING_PROFILE[t]}' for t in TIER_ORDER)
    ax.text(0.028, 0.985,
            f'{len(profiles)} profiles  $\\rightarrow$  '
            f'{len(signatures)} sub-networks\n'
            f'Ceiling:  {ceiling}',
            transform=ax.transAxes, fontsize=FS_ANNOT, color=INK, va='top',
            ha='left', linespacing=1.45,
            bbox=dict(boxstyle='round,pad=0.34', facecolor='#fcfcfb',
                      edgecolor=AXIS, linewidth=0.7))

    ax.set_xlim(0.55, 1.06)
    ax.set_ylim(0.12, 1.06)
    ax.set_xlabel('layer3 capacity', fontsize=FS_LABEL, color=INK,
                  fontweight='bold')
    ax.set_ylabel('layer4 capacity', fontsize=FS_LABEL, color=INK,
                  fontweight='bold')
    ax.set_title('(b) Allocations form a continuum, not 5 points',
                 fontsize=FS_TITLE, color=INK, pad=6, fontweight='bold')


def style_axes(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=FS_TICK, colors=MUTED, length=3.5, width=0.9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK)
        label.set_fontweight('bold')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', required=True)
    parser.add_argument('--out', default='figures/liquid_dynamics')
    arguments = parser.parse_args()

    traj, manifest = load_trace(arguments.trace)
    profiles = manifest['profile_capacities']
    model_name = manifest['model']
    num_classes = 100 if manifest['dataset'] == 'cifar100' else 10
    cost_fn = build_cost_fn(manifest.get('arguments', {}), model_name, num_classes)

    # tier budgets = cost of each tier's ceiling profile (F1 / F3 / F4)
    tier_budget = {
        'weak': cost_fn(profiles[1]),
        'medium': cost_fn(profiles[3]),
        'strong': cost_fn(profiles[4]),
    }
    print('tier budgets:', {k: round(v, 4) for k, v in tier_budget.items()})
    print('personalized clients:', len(traj))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    fig.patch.set_facecolor('#fcfcfb')
    for ax in axes:
        ax.set_facecolor('#fcfcfb')

    panel_a(axes[0], traj, tier_budget)
    panel_b(axes[1], traj, cost_fn, profiles, tier_budget)
    for ax in axes:
        style_axes(ax)

    handles = [Line2D([], [], color=TIER_COLOR[t], lw=1.8, label=TIER_LABEL[t])
               for t in TIER_ORDER]
    handles.append(Line2D([], [], color=MUTED, marker='o', ls='none', ms=4.6,
                          mec='#fcfcfb', mew=1.0,
                          label='Trigger event (structure updated)'))
    handles.append(Line2D([], [], color=INK, marker='D', ls='none', ms=4.0,
                          label='Fixed profile anchor'))
    legend = fig.legend(handles=handles, loc='lower center', ncol=5,
                        frameon=False, fontsize=FS_LEGEND,
                        bbox_to_anchor=(0.5, 0.012),
                        handlelength=1.7, columnspacing=1.5)
    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_color(INK)

    fig.tight_layout(rect=(0, 0.105, 1, 1), w_pad=2.4)
    os.makedirs(os.path.dirname(arguments.out) or '.', exist_ok=True)
    for extension in ('pdf', 'png'):
        path = f'{arguments.out}.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='#fcfcfb')
        print('wrote', path)


if __name__ == '__main__':
    main()
