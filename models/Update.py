#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import random
from optimizer.Adabelief import AdaBelief


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, args):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.model = args.model

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if "transformer" in self.model:
            image, label, mask = self.dataset[self.idxs[item]].values()
            return image, label, mask
        else:
            image, label = self.dataset[self.idxs[item]]
            return image, label


class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()

        self.kld_loss = nn.KLDivLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.T = 3
        self.gamma = 0

    def loss_fn_kd(self, pred, target, soft_target, gamma_active=True):
        _ce = self.ce_loss(pred, target)
        T = self.T
        if self.gamma and gamma_active:
            # _ce = (1. - self.gamma) * _ce
            _kld = self.kld_loss(self.log_softmax(pred / T), self.softmax(soft_target / T)) * self.gamma * T * T
        else:
            _kld = 0
        loss = _ce + _kld
        return loss


class LocalUpdate_ScaleFL(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = KDLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, args), batch_size=self.args.local_bs, shuffle=True,
                                    drop_last=True)
        self.verbose = verbose

    def train(self, round, net, ee):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr * (self.args.lr_decay ** round),
                                        momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if self.args.dataset == 'widar':
                    labels = labels.long()
                net.zero_grad()
                outputs = net(images, ee)
                loss = 0.0
                for j in range(len(outputs)):
                    if j == len(outputs) - 1:
                        loss += self.loss_func.ce_loss(outputs[j]["output"], labels) * (j + 1)
                    else:
                        gamma_active = round > self.args.epochs * 0.25
                        loss += self.loss_func.loss_fn_kd(outputs[j]["output"], labels, outputs[-1]["output"], gamma_active) * (j + 1)

                loss /= len(outputs) * (len(outputs) + 1) / 2

                loss.backward()
                optimizer.step()

                Predict_loss += loss.item()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        requires_grad = []
        for param in net.parameters():
            requires_grad.append(param.grad != None)

        return net.state_dict(), requires_grad


class LocalUpdate_FedAvg(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, args), batch_size=self.args.local_bs, shuffle=True,
                                    drop_last=True)
        self.verbose = verbose

    def train(self, round, net):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr * (self.args.lr_decay ** round),
                                        momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        for iter in range(self.args.local_ep):
            if "transformer" in self.args.model:
                for batch_idx, (images, labels, mask) in enumerate(self.ldr_train):
                    images, labels, mask = images.to(self.args.device), labels.to(self.args.device), mask.to(
                        self.args.device)
                    net.zero_grad()
                    log_probs = net(images, mask)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                    Predict_loss += loss.item()
            else:
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    if self.args.dataset == 'widar':
                        labels = labels.long()
                    net.zero_grad()
                    log_probs = net(images)['output']
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                    Predict_loss += loss.item()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net.state_dict()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def _rep_to_vec(rep: torch.Tensor) -> torch.Tensor:
    # 兼容 [B, D] 或 [B, C, H, W]
    if rep.dim() == 4:
        rep = F.adaptive_avg_pool2d(rep, 1).flatten(1)
    elif rep.dim() == 2:
        pass
    else:
        raise ValueError(f"Unexpected representation shape: {tuple(rep.shape)}")
    return rep


def paper_supervised_objective(task_logits, measurement_embedding, labels,
                               measurement_classifier, criterion=None):
    """Paper-aligned supervised objective for backbone, adapter and classifier.

    Both terms are ordinary label-supervised classification.  No prototype,
    anchor, cross-client matching, or alignment loss is introduced.
    """
    criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    task_loss = criterion(task_logits, labels)
    measurement_loss = criterion(measurement_classifier(measurement_embedding), labels)
    return task_loss + measurement_loss


class LocalUpdate_AdaptiveFL(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False, client_id=None, num_classes=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, args),
            batch_size=self.args.local_bs,
            shuffle=True,
            drop_last=True
        )
        self.verbose = verbose
        self.idxs = idxs
        self.client_id = client_id
        # num_classes 优先用显式传入，其次 args.num_classes（如果你 args 里有）
        self.num_classes = int(num_classes) if num_classes is not None else int(getattr(args, "num_classes", 10))

    # ======== 你原来的 AdaptiveFL 本地训练：完全保留不动 ========
    def train(self, round, net):
        net.train()
        prox_setup = getattr(self, 'adapterfl_prox', None)
        if prox_setup is not None:
            from adapterfl_methods import prepare_prox, prox_loss, prox_gradient_diagnostics
            prox_targets, prox_count, prox_strength = prepare_prox(net, prox_setup)
            prox_normalization = prox_setup[2] if len(prox_setup)>2 else 'mean'
            prox_denominator = prox_count if prox_normalization=='mean' else 1
            task_sum, prox_sum, steps = 0.0, 0.0, 0
            prox_samples, ratio_sum = [], 0.0
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                net.parameters(),
                lr=self.args.lr * (self.args.lr_decay ** round),
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")

        Predict_loss = 0.0
        for _ in range(self.args.local_ep):
            for _, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if self.args.dataset == 'widar':
                    labels = labels.long()

                net.zero_grad()
                log_probs = net(images)["output"]
                loss = self.loss_func(log_probs, labels)
                if prox_setup is not None:
                    regularizer = prox_loss(net, prox_targets, prox_count, prox_normalization)
                    task_sum += float(loss.detach())
                    prox_sum += float(regularizer.detach())
                    steps += 1
                    ratio_sum += float((prox_strength * regularizer.detach()) / loss.detach().clamp_min(1e-12))
                    if steps <= getattr(self, 'adapterfl_prox_diagnostic_steps', 2):
                        prox_samples.append(dict(prox_gradient_diagnostics(net,loss,regularizer,prox_targets,prox_count,prox_strength,prox_normalization),step=steps))
                    loss = loss + prox_strength * regularizer
                loss.backward()
                optimizer.step()
                Predict_loss += float(loss.item())

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        if prox_setup is not None:
            self.adapterfl_metrics = {'task_ce_mean': task_sum / max(steps, 1),
                                     'prox_loss_mean': prox_sum / max(steps, 1),
                                     'lambda_prox': prox_strength if prox_normalization=='mean' else None,
                                     'mu_prox': prox_strength / prox_denominator,
                                     'prox_normalization': prox_normalization,'target_parameters': prox_count,
                                     'effective_coefficient_per_parameter': prox_strength / prox_denominator,
                                     'weighted_prox_loss_mean': prox_strength * prox_sum / max(steps, 1),
                                     'weighted_prox_to_task_loss_ratio_mean': ratio_sum / max(steps, 1),
                                     'gradient_samples': prox_samples,
                                     'diagnostic_extra_task_backwards': len(prox_samples),
                                     'steps': steps}
            valid = [r['weighted_prox_to_task_shared_gradient_ratio'] for r in prox_samples
                     if r['proximal_gradient_l2'] > 0 and r['weighted_prox_to_task_shared_gradient_ratio'] is not None]
            self.adapterfl_metrics['warning_codes'] = ['PROX_GRADIENT_BELOW_1E_4_OF_TASK_SHARED_GRADIENT'] if valid and max(valid)<1e-4 else []
        return net.state_dict()

    # ======== 新增：SemanticFL 本地训练（与 client.py 同逻辑）========
    def train_with_proto(self, round, net, semantic_server):
        """
        返回:
          w (state_dict),
          n_samples,
          avg_loss,
          protos: {class_id: tensor_cpu},
          counts: {class_id: int}
        """
        if semantic_server is None:
            raise ValueError("semantic_server must be provided for train_with_proto().")
        if self.client_id is None:
            raise ValueError("client_id must be provided for train_with_proto().")

        net.train()

        # optimizer：保持你原来的策略，但要能把 adapter/sem_clf 动态加进去
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                list(net.parameters()),
                lr=self.args.lr * (self.args.lr_decay ** round),
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(list(net.parameters()), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(list(net.parameters()), lr=self.args.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")

        adapter = None
        sem_clf = None

        epoch_losses = []

        for _ in range(self.args.local_ep):
            batch_losses = []
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if self.args.dataset == 'widar':
                    labels = labels.long()

                out = net(images)
                logits = out["output"]
                rep = _rep_to_vec(out["representation"])

                # 懒初始化 adapter / sem_clf，并加入 optimizer param_group
                if adapter is None:
                    adapter = semantic_server.get_adapter(self.client_id, rep.size(1)).to(self.args.device)
                    optimizer.add_param_group({"params": adapter.parameters()})
                    sem_clf = semantic_server.get_sem_clf(self.client_id, self.num_classes).to(self.args.device)
                    optimizer.add_param_group({"params": sem_clf.parameters()})

                # head 冻结 + eval（SemanticFL 约定）
                semantic_server.head.eval()
                # 这里 head 是 no_grad；但 adapter 需要梯度，所以重新算一遍（避免 adapter 没梯度）
                z = semantic_server.head(adapter(rep))
                z = F.normalize(z, p=2, dim=1)

                loss = paper_supervised_objective(
                    logits, z, labels, sem_clf, criterion=self.loss_func
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                batch_losses.append(float(loss.item()))

            if batch_losses:
                epoch_losses.append(sum(batch_losses) / len(batch_losses))

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        # Prototypes are measurements only; they never enter the local loss.
        if adapter is None:
            return net.state_dict(), len(self.idxs), avg_loss, {}, {}

        net.eval()
        adapter.eval()
        semantic_server.head.eval()

        sums = {}
        counts = {}
        with torch.no_grad():
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if self.args.dataset == 'widar':
                    labels = labels.long()

                out = net(images)
                rep = _rep_to_vec(out["representation"])
                z = semantic_server.head(adapter(rep))
                z = F.normalize(z, p=2, dim=1)

                for i in range(z.size(0)):
                    c = int(labels[i].item())
                    if c not in sums:
                        sums[c] = torch.zeros(z.size(1), device=z.device)
                        counts[c] = 0
                    sums[c] += z[i]
                    counts[c] += 1

        protos = {
            c: F.normalize(sums[c] / max(1, counts[c]), p=2, dim=0).detach().cpu()
            for c in sums
        }
        counts = {c: int(counts[c]) for c in counts}

        return net.state_dict(), len(self.idxs), avg_loss, protos, counts



class LocalUpdate_FedProx(object):
    def __init__(self, args, glob_model, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True,
                                    drop_last=True)
        self.glob_model = glob_model
        self.prox_alpha = args.prox_alpha
        self.verbose = verbose

    def train(self, round, net):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr * (self.args.lr_decay ** round),
                                        momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        Penalize_loss = 0

        global_weight_collector = list(self.glob_model.parameters())

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)['output']
                predictive_loss = self.loss_func(log_probs, labels)

                # for fedprox
                fed_prox_reg = 0.0
                # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += (
                            (self.prox_alpha / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)

                loss = predictive_loss + fed_prox_reg
                Predict_loss += predictive_loss.item()
                Penalize_loss += fed_prox_reg.item()

                loss.backward()
                optimizer.step()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            info += ', Penalize loss={:.4f}'.format(Penalize_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net.state_dict()
