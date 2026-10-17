#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local optimization for clients with immutable submodel structures."""

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.Fed import split_model
from models.Update import DatasetSplit


def resolve_kd_decay_window(args):
    """Resolve explicit rounds or the default 30%-80% KD decay window."""
    start = int(getattr(args, 'kd_decay_start', -1))
    end = int(getattr(args, 'kd_decay_end', -1))
    if start < 0 and end < 0:
        epochs = max(int(getattr(args, 'epochs', 1)), 1)
        start = int(round(0.30 * max(epochs - 1, 0)))
        end = int(round(0.80 * max(epochs - 1, 0)))
        end = max(end, start + 1)
    elif start < 0 or end < 0:
        raise ValueError(
            'kd_decay_start/end must both be -1 or both be nonnegative')
    elif end <= start:
        raise ValueError('kd_decay_end must exceed kd_decay_start')
    return start, end


def kd_gamma_for_round(args, round_idx):
    """Cosine-decay the nested-subnetwork distillation coefficient."""
    gamma = float(getattr(args, 'gamma', 0.0))
    if gamma <= 0.0:
        return 0.0
    start, end = resolve_kd_decay_window(args)
    final_ratio = float(getattr(args, 'kd_final_ratio', 0.1))
    if not 0.0 <= final_ratio <= 1.0:
        raise ValueError('kd_final_ratio must lie in [0, 1]')
    if round_idx <= start:
        return gamma
    if round_idx >= end:
        return gamma * final_ratio
    progress = (float(round_idx) - start) / max(float(end - start), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return gamma * (final_ratio + (1.0 - final_ratio) * cosine)


class LocalUpdate_StaticSubmodel:
    """CE + nested self-distillation under a fixed client shape."""

    def __init__(self, args, dataset, idxs, verbose=False):
        self.args = args
        self.verbose = bool(verbose)
        self.indices = list(idxs)
        if not self.indices:
            raise ValueError('a client cannot train on an empty dataset')
        batch_size = max(1, min(int(args.local_bs), len(self.indices)))
        self.loader = DataLoader(
            DatasetSplit(dataset, self.indices, args),
            batch_size=batch_size, shuffle=True, drop_last=True)
        self.loss_function = nn.CrossEntropyLoss()

    def _optimizer(self, net, round_idx):
        if self.args.optimizer == 'adam':
            return torch.optim.Adam(net.parameters(), lr=self.args.lr)
        if self.args.optimizer != 'sgd':
            raise ValueError('optimizer must be sgd or adam')
        return torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr * (self.args.lr_decay ** round_idx),
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay)

    def train(self, round_idx, net, teacher_models=None):
        net.train()
        optimizer = self._optimizer(net, round_idx)
        kd_gamma = kd_gamma_for_round(self.args, round_idx)
        temperature = float(getattr(self.args, 'T', 3.0))
        if temperature <= 0.0:
            raise ValueError('T must be positive')
        teachers = list(teacher_models or []) if kd_gamma > 0.0 else []
        for teacher in teachers:
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)

        ce_losses, kd_losses, total_losses, grad_norms = [], [], [], []
        train_correct = 0
        train_total = 0
        steps = 0
        processed_samples = 0
        active_params = max(sum(p.numel() for p in net.parameters()), 1)

        for _ in range(int(self.args.local_ep)):
            for batch in self.loader:
                images, labels = batch[:2]
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                if self.args.dataset == 'widar':
                    labels = labels.long()

                optimizer.zero_grad(set_to_none=True)
                logits = net(images)['output']
                ce_loss = self.loss_function(logits, labels)
                kd_loss = ce_loss.new_zeros(())
                if teachers:
                    student_log_prob = F.log_softmax(
                        logits / temperature, dim=1)
                    teacher_probabilities = []
                    student_state = net.state_dict()
                    with torch.no_grad():
                        for teacher in teachers:
                            teacher.load_state_dict(
                                split_model(student_state,
                                            teacher.state_dict()),
                                strict=True)
                            teacher_logits = teacher(images)['output']
                            teacher_probabilities.append(F.softmax(
                                teacher_logits / temperature, dim=1))
                    for probability in teacher_probabilities:
                        kd_loss = kd_loss + F.kl_div(
                            student_log_prob, probability,
                            reduction='batchmean') / logits.shape[1]
                    kd_loss = (kd_loss / len(teacher_probabilities)
                               * temperature * temperature)

                loss = ce_loss + kd_gamma * kd_loss
                loss.backward()
                grad_sq = loss.new_zeros(())
                for parameter in net.parameters():
                    if parameter.grad is not None:
                        grad_sq += parameter.grad.detach().square().sum()
                grad_norms.append(float(
                    (grad_sq.sqrt() / (active_params ** 0.5)).item()))
                optimizer.step()

                ce_losses.append(float(ce_loss.item()))
                kd_losses.append(float(kd_loss.item()))
                total_losses.append(float(loss.item()))
                train_correct += int((logits.argmax(dim=1) == labels).sum())
                train_total += int(labels.size(0))
                processed_samples += int(labels.size(0))
                steps += 1

        stats = {
            'train_acc': train_correct / max(train_total, 1),
            'mean_loss': float(np.mean(ce_losses)) if ce_losses else 0.0,
            'mean_kd_loss': (
                float(np.mean(kd_losses)) if kd_losses else 0.0),
            'mean_total_loss': (
                float(np.mean(total_losses)) if total_losses else 0.0),
            'kd_gamma': kd_gamma,
            'teacher_count': len(teachers),
            'loss_drop': (
                float(ce_losses[0] - ce_losses[-1]) if ce_losses else 0.0),
            'grad_norm': (
                float(np.mean(grad_norms)) if grad_norms else 0.0),
            'loss_var': float(np.var(ce_losses)) if ce_losses else 0.0,
            'steps': steps,
            'processed_samples': processed_samples,
        }
        if self.verbose:
            print('local loss={:.4f} kd={:.4f} acc={:.4f}'.format(
                stats['mean_loss'], stats['mean_kd_loss'],
                stats['train_acc']))
        return net.state_dict(), stats


__all__ = [
    'LocalUpdate_StaticSubmodel', 'kd_gamma_for_round',
    'resolve_kd_decay_window',
]
