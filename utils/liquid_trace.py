#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auditable trace collection for the LiquidFL motivation experiment.

The recorder is deliberately passive: it serializes already-computed
controller state and never changes client sampling, model dispatch, gradients,
or aggregation.  Event records are appended once per participating client,
using that client's participation count as the local time axis.
"""

import json
import os

import numpy as np
import torch


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(
            _json_value(payload), handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _dataset_targets(dataset):
    targets = getattr(dataset, 'targets', None)
    if targets is None:
        targets = getattr(dataset, 'labels', None)
    if targets is None:
        return None
    return np.asarray(targets, dtype=np.int64)


def _class_histogram(targets, indices, num_classes):
    if targets is None:
        return []
    values = targets[np.asarray(list(indices), dtype=np.int64)]
    return np.bincount(values, minlength=int(num_classes)).astype(int).tolist()


def _numeric_summary(values):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return {
            'count': 0, 'mean': None, 'std': None, 'min': None,
            'p10': None, 'median': None, 'p90': None, 'max': None,
        }
    return {
        'count': int(values.size),
        'mean': float(values.mean()),
        'std': float(values.std()),
        'min': float(values.min()),
        'p10': float(np.quantile(values, 0.10)),
        'median': float(np.quantile(values, 0.50)),
        'p90': float(np.quantile(values, 0.90)),
        'max': float(values.max()),
    }


def _weighted_local_metrics(tasks):
    if not tasks:
        return {}
    weights = np.asarray(
        [max(int(task.get('n', 0)), 1) for task in tasks],
        dtype=np.float64)
    weight_sum = max(float(weights.sum()), 1.0)
    keys = (
        'train_acc', 'mean_loss', 'mean_kd_loss', 'mean_total_loss',
        'loss_drop', 'grad_norm', 'loss_var', 'teacher_count', 'steps',
        'processed_samples',
    )
    result = {}
    for key in keys:
        values = np.asarray(
            [float(task.get('stats', {}).get(key, 0.0)) for task in tasks],
            dtype=np.float64)
        result[key] = float(np.dot(weights, values) / weight_sum)
    return result


def _tier_capacity_summary(states, projector):
    result = {}
    for tier in ('weak', 'medium', 'strong'):
        tier_states = [
            state for _, state in sorted(states.states.items())
            if state.tier == tier and state.capacity is not None
        ]
        if not tier_states:
            result[tier] = {
                'active_clients': 0,
                'stage_capacity': [],
                'cost_ratio': _numeric_summary([]),
                'budget_utilization': _numeric_summary([]),
                'fast_memory_norm': _numeric_summary([]),
                'slow_memory_norm': _numeric_summary([]),
            }
            continue
        capacities = np.asarray([
            state.capacity.detach().cpu().reshape(3).tolist()
            for state in tier_states
        ], dtype=np.float64)
        costs = [
            float(projector.exact_cost(state.capacity))
            for state in tier_states
        ]
        budget_utilization = [
            cost / float(state.budget)
            for state, cost in zip(tier_states, costs)
            if state.budget
        ]
        result[tier] = {
            'active_clients': len(tier_states),
            'stage_capacity': [
                _numeric_summary(capacities[:, stage])
                for stage in range(capacities.shape[1])
            ],
            'cost_ratio': _numeric_summary(costs),
            'budget_utilization': _numeric_summary(budget_utilization),
            'fast_memory_norm': _numeric_summary([
                float(state.fast_memory.norm(p=2).item())
                for state in tier_states
                if state.fast_memory is not None
            ]),
            'slow_memory_norm': _numeric_summary([
                float(state.slow_memory.norm(p=2).item())
                for state in tier_states
                if state.slow_memory is not None
            ]),
        }
    return result


def build_client_holdout(dataset, dict_users, num_classes, val_ratio, seed):
    """Create a deterministic client-local train/validation split.

    This helper is used only by the dedicated motivation entry point.  The
    validation indices are never passed to LiquidFL training.
    """
    ratio = float(val_ratio)
    if not 0.0 < ratio < 0.5:
        raise ValueError('motivation validation ratio must be in (0, 0.5)')
    targets = _dataset_targets(dataset)
    train_users, val_users, clients = {}, {}, {}
    for client_id in sorted(int(cid) for cid in dict_users):
        indices = np.asarray(
            list(dict_users[client_id]), dtype=np.int64)
        if indices.size < 2:
            raise ValueError(
                'client {} has fewer than two samples and cannot be split'
                .format(client_id))
        rng = np.random.RandomState(
            int(seed) * 1000003 + int(client_id) * 9176 + 73)
        indices = indices[rng.permutation(indices.size)]
        val_count = min(
            max(int(round(indices.size * ratio)), 1), indices.size - 1)
        val_indices = indices[:val_count].astype(int).tolist()
        train_indices = indices[val_count:].astype(int).tolist()
        train_users[client_id] = train_indices
        val_users[client_id] = val_indices
        clients[str(client_id)] = {
            'client_id': client_id,
            'total_samples': int(indices.size),
            'train_samples': len(train_indices),
            'val_samples': len(val_indices),
            'full_label_hist': _class_histogram(
                targets, indices, num_classes),
            'train_label_hist': _class_histogram(
                targets, train_indices, num_classes),
            'val_label_hist': _class_histogram(
                targets, val_indices, num_classes),
            'train_indices': train_indices,
            'val_indices': val_indices,
        }
    return train_users, val_users, clients


def write_holdout_manifest(path, args, clients):
    _atomic_json(path, {
        'schema_version': 1,
        'purpose': 'client-local held-out evaluation; not used by training',
        'dataset': args.dataset,
        'data_beta': float(args.data_beta),
        'seed': int(args.seed),
        'num_classes': int(args.num_classes),
        'clients': clients,
    })


def _state_payload(state, projector):
    capacity = (
        state.capacity.detach().cpu().tolist()
        if state.capacity is not None else None)
    cost = (
        float(projector.exact_cost(state.capacity))
        if state.capacity is not None else None)
    forward_macs = (
        float(projector.estimate_forward_macs(state.capacity))
        if state.capacity is not None else None)
    return {
        'client_id': int(state.client_id),
        'tier': str(state.tier),
        'participation_count': int(state.participation_count),
        'last_participation_round': int(state.last_participation_round),
        'personalized': bool(state.personalized),
        'anchor_profile': (
            int(state.anchor_profile)
            if state.anchor_profile is not None else None),
        'promotion_round': int(state.promotion_round),
        'capacity': capacity,
        'hard_signature': (
            list(projector.hard_signature(state.capacity))
            if state.capacity is not None else None),
        'cost_ratio': cost,
        'forward_macs': forward_macs,
        'budget': (
            float(state.budget) if state.budget is not None else None),
        'budget_utilization': (
            cost / float(state.budget)
            if cost is not None and state.budget else None),
        'fixed_mask': (
            state.fixed_mask.detach().cpu().tolist()
            if state.fixed_mask is not None else None),
        'trajectory': state.trajectory.detach().cpu().tolist(),
        'stagnation_ema': float(state.stagnation_ema),
        'need_streak': int(state.need_streak),
        'last_need': float(state.last_need),
        'last_gate_open': bool(state.last_gate_open),
        'personalized_uses': int(state.personalized_uses),
        'reuse_uses': int(state.reuse_uses),
        'exploration_uses': int(state.exploration_uses),
    }


class LiquidTraceRecorder:
    """Incremental JSONL trace plus restart-safe state/checkpoint snapshots."""

    def __init__(self, args, dataset, dict_users, states, projector,
                 profile_capacities):
        self.enabled = bool(int(getattr(args, 'liquid_trace_enabled', 0)))
        self.root = None
        if not self.enabled:
            return
        configured = str(getattr(args, 'liquid_trace_dir', '')).strip()
        if configured:
            self.root = os.path.abspath(configured)
        else:
            stamp = str(getattr(args, 'run_stamp', 'unstamped'))
            self.root = os.path.abspath(os.path.join(
                getattr(args, 'output_dir', './output'),
                'liquid_trace_' + stamp))
        os.makedirs(self.root, exist_ok=True)
        self.events_path = os.path.join(self.root, 'client_events.jsonl')
        self.server_path = os.path.join(self.root, 'server_rounds.jsonl')
        self.states_path = os.path.join(self.root, 'latest_states.json')
        self.checkpoint_path = os.path.join(
            self.root, 'latest_checkpoint.pt')
        self.checkpoint_interval = max(
            int(getattr(args, 'liquid_trace_checkpoint_interval', 0)), 0)

        targets = _dataset_targets(dataset)
        clients = {}
        for client_id in sorted(int(cid) for cid in dict_users):
            state = states.get(client_id)
            clients[str(client_id)] = {
                'client_id': client_id,
                'tier': state.tier,
                'tier_budget': float(states.tier_budget[state.tier]),
                'training_samples': len(dict_users[client_id]),
                'training_label_hist': _class_histogram(
                    targets, dict_users[client_id], args.num_classes),
            }
        _atomic_json(os.path.join(self.root, 'trace_manifest.json'), {
            'schema_version': 2,
            'description': (
                'Passive client/server LiquidFL trajectory instrumentation'),
            'artifacts': {
                'client_events': 'client_events.jsonl',
                'server_rounds': 'server_rounds.jsonl',
                'latest_states': 'latest_states.json',
                'latest_checkpoint': 'latest_checkpoint.pt',
            },
            'dataset': args.dataset,
            'model': args.model,
            'data_beta': float(args.data_beta),
            'seed': int(args.seed),
            'num_clients': int(args.num_users),
            'capacity_dimension': 3,
            'capacity_semantics': ['layer2', 'layer3', 'layer4'],
            'profile_capacities': profile_capacities,
            'arguments': vars(args),
            'clients': clients,
        })
        print('[LiquidTrace] artifacts -> {}'.format(self.root))

    def _snapshot(self, round_idx, states, projector):
        if not self.enabled:
            return
        _atomic_json(self.states_path, {
            'schema_version': 1,
            'round': int(round_idx),
            'clients': {
                str(client_id): _state_payload(state, projector)
                for client_id, state in sorted(states.states.items())
            },
        })

    def record_round(self, round_idx, states, projector, plan,
                     profiles_by_client, tasks_by_client,
                     controller_loss, controller_grad_norm, q_scale,
                     round_summary=None):
        if not self.enabled:
            return
        rows = []
        for client_id in sorted(tasks_by_client):
            state = states.get(client_id)
            decision = plan[client_id]
            tasks = tasks_by_client[client_id]
            q_feedback = decision.get('q_feedback')
            capacity_before = decision.get('base_capacity')
            capacity_after = state.capacity
            capacity_delta = None
            if capacity_before is not None and capacity_after is not None:
                capacity_delta = (
                    capacity_after.detach().cpu()
                    - torch.as_tensor(capacity_before).detach().cpu()
                ).tolist()
            q_tensor = (
                torch.as_tensor(q_feedback, dtype=torch.float32).reshape(3)
                if q_feedback is not None else None)
            delta_tensor = (
                torch.as_tensor(capacity_delta, dtype=torch.float32).reshape(3)
                if capacity_delta is not None else None)
            first_order_gain = None
            feedback_alignment = None
            if q_tensor is not None and delta_tensor is not None:
                first_order_gain = float(
                    -torch.dot(q_tensor, delta_tensor).item())
                denominator = float(
                    q_tensor.norm(p=2).item()
                    * delta_tensor.norm(p=2).item())
                if denominator > 1e-12:
                    feedback_alignment = float(
                        -torch.dot(q_tensor, delta_tensor).item()
                        / denominator)
            task_capacities = [
                task['capacity'].detach().cpu().tolist() for task in tasks]
            dispatched = [
                task['dispatched_capacity'].detach().cpu().tolist()
                for task in tasks]
            rows.append({
                'schema_version': 2,
                'round': int(round_idx),
                'client_id': int(client_id),
                'tier': state.tier,
                'participation_event': int(state.participation_count),
                'sample_count': int(sum(task['n'] for task in tasks)),
                'dispatched_profiles': [
                    int(value) for value in profiles_by_client[client_id]],
                'dispatched_capacities': dispatched,
                'task_capacities': task_capacities,
                'dispatched_cost_ratios': [
                    float(projector.exact_cost(
                        task['dispatched_capacity']))
                    for task in tasks
                ],
                'task_cost_ratios': [
                    float(projector.exact_cost(task['capacity']))
                    for task in tasks
                ],
                'dispatched_parameter_counts': [
                    int(projector.exact_parameter_count(
                        task['dispatched_capacity']))
                    for task in tasks
                ],
                'task_parameter_counts': [
                    int(projector.exact_parameter_count(task['capacity']))
                    for task in tasks
                ],
                'local_metrics': _weighted_local_metrics(tasks),
                'control_event': bool(decision.get('control', False)),
                'control_profile': (
                    int(decision['control_profile'])
                    if decision.get('control_profile') is not None else None),
                'raw_need': bool(decision.get('raw_need', False)),
                'gate_open': bool(decision.get('gate_open', False)),
                'promoted': bool(decision.get('promoted', False)),
                'reuse_gate': bool(decision.get('reuse_gate', False)),
                'exploration': bool(decision.get('exploration', False)),
                'growth_phase': bool(decision.get('growth_phase', False)),
                'signature_changed': bool(
                    decision.get('sig_changed', False)),
                'probe_loss': decision.get('probe'),
                'z_self': float(decision.get('z_self', 0.0)),
                'z_pool': float(decision.get('z_pool', 0.0)),
                'stagnation_ema': float(state.stagnation_ema),
                'capacity_before': _json_value(capacity_before),
                'proposal_capacity': _json_value(
                    decision.get('proposal_capacity')),
                'capacity_after': _json_value(capacity_after),
                'capacity_delta': capacity_delta,
                'capacity_delta_l2': (
                    float(delta_tensor.norm(p=2).item())
                    if delta_tensor is not None else None),
                'capacity_delta_linf': (
                    float(delta_tensor.abs().max().item())
                    if delta_tensor is not None else None),
                'proposal_cost_ratio': (
                    float(projector.exact_cost(
                        decision['proposal_capacity']))
                    if decision.get('proposal_capacity') is not None
                    else None),
                'hard_signature_after': (
                    list(projector.hard_signature(capacity_after))
                    if capacity_after is not None else None),
                'cost_ratio_after': (
                    float(projector.exact_cost(capacity_after))
                    if capacity_after is not None else None),
                'forward_macs_after': (
                    float(projector.estimate_forward_macs(capacity_after))
                    if capacity_after is not None else None),
                'budget': (
                    float(state.budget)
                    if state.budget is not None else None),
                'budget_utilization': (
                    float(projector.exact_cost(capacity_after))
                    / float(state.budget)
                    if capacity_after is not None and state.budget else None),
                'anchor_profile': (
                    int(state.anchor_profile)
                    if state.anchor_profile is not None else None),
                'fixed_mask': _json_value(state.fixed_mask),
                'trajectory': state.trajectory.detach().cpu().tolist(),
                'fast_memory_norm': (
                    float(state.fast_memory.norm(p=2).item())
                    if state.fast_memory is not None else None),
                'slow_memory_norm': (
                    float(state.slow_memory.norm(p=2).item())
                    if state.slow_memory is not None else None),
                'q_feedback': _json_value(q_feedback),
                'stage_grad_norm': (
                    torch.as_tensor(q_feedback).abs().cpu().tolist()
                    if q_feedback is not None else None),
                'q_norm': float(decision.get('q_norm', 0.0)),
                'first_order_predicted_gain': first_order_gain,
                'feedback_action_alignment': feedback_alignment,
                'controller_loss': controller_loss,
                'controller_grad_norm': float(controller_grad_norm),
                'q_scale': float(q_scale),
            })
        with open(self.events_path, 'a', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(
                    _json_value(row), ensure_ascii=False) + '\n')
            handle.flush()

        all_tasks = [
            task for tasks in tasks_by_client.values() for task in tasks]
        dispatched_capacities = [
            task['dispatched_capacity'].detach().cpu().reshape(3).tolist()
            for task in all_tasks]
        actual_capacities = [
            task['capacity'].detach().cpu().reshape(3).tolist()
            for task in all_tasks]
        dispatched_parameters = [
            int(projector.exact_parameter_count(
                task['dispatched_capacity']))
            for task in all_tasks]
        actual_parameters = [
            int(projector.exact_parameter_count(task['capacity']))
            for task in all_tasks]
        dispatch_counts = {}
        for profile_indices in profiles_by_client.values():
            for profile_idx in profile_indices:
                key = 'F{}'.format(int(profile_idx))
                dispatch_counts[key] = dispatch_counts.get(key, 0) + 1
        q_norms = [
            float(decision.get('q_norm', 0.0))
            for decision in plan.values()
            if decision.get('q_feedback') is not None
        ]
        action_norms = []
        action_alignments = []
        predicted_gains = []
        for row in rows:
            if row['capacity_delta_l2'] is not None:
                action_norms.append(row['capacity_delta_l2'])
            if row['feedback_action_alignment'] is not None:
                action_alignments.append(row['feedback_action_alignment'])
            if row['first_order_predicted_gain'] is not None:
                predicted_gains.append(row['first_order_predicted_gain'])
        actual_parameter_sum = int(sum(actual_parameters))
        dispatched_parameter_sum = int(sum(dispatched_parameters))
        server_row = {
            'schema_version': 2,
            'round': int(round_idx),
            'round_summary': _json_value(round_summary or {}),
            'selected_tasks': len(all_tasks),
            'selected_unique_clients': len(tasks_by_client),
            'selected_samples': int(sum(
                int(task.get('n', 0)) for task in all_tasks)),
            'dispatch_counts': dispatch_counts,
            'dispatched_stage_capacity': [
                _numeric_summary(
                    np.asarray(dispatched_capacities)[:, stage]
                    if dispatched_capacities else [])
                for stage in range(3)
            ],
            'actual_stage_capacity': [
                _numeric_summary(
                    np.asarray(actual_capacities)[:, stage]
                    if actual_capacities else [])
                for stage in range(3)
            ],
            'dispatched_parameter_sum': dispatched_parameter_sum,
            'actual_parameter_sum': actual_parameter_sum,
            'extra_parameter_sum': (
                actual_parameter_sum - dispatched_parameter_sum),
            'model_payload_ratio_vs_dispatch': (
                actual_parameter_sum / dispatched_parameter_sum
                if dispatched_parameter_sum > 0 else None),
            'q_norm': _numeric_summary(q_norms),
            'capacity_action_l2': _numeric_summary(action_norms),
            'feedback_action_alignment': _numeric_summary(
                action_alignments),
            'first_order_predicted_gain': _numeric_summary(predicted_gains),
            'tier_capacity_state': _tier_capacity_summary(
                states, projector),
        }
        with open(self.server_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(
                _json_value(server_row), ensure_ascii=False) + '\n')
            handle.flush()
        self._snapshot(round_idx, states, projector)

    def maybe_checkpoint(self, round_idx, final_round, global_model, states,
                         projector, profile_capacities, args):
        if not self.enabled:
            return
        due = bool(final_round)
        if self.checkpoint_interval > 0:
            due = due or ((int(round_idx) + 1) % self.checkpoint_interval == 0)
        if not due:
            return
        payload = {
            'schema_version': 1,
            'round': int(round_idx),
            'model_state': {
                key: value.detach().cpu().clone()
                for key, value in global_model.state_dict().items()
            },
            'client_states': {
                str(client_id): _state_payload(state, projector)
                for client_id, state in sorted(states.states.items())
            },
            'profile_capacities': _json_value(profile_capacities),
            'arguments': _json_value(vars(args)),
        }
        temporary = self.checkpoint_path + '.tmp'
        torch.save(payload, temporary)
        os.replace(temporary, self.checkpoint_path)
        print('[LiquidTrace] checkpoint round={} -> {}'.format(
            round_idx, self.checkpoint_path))
