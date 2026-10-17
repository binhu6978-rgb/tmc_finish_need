#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import random
from collections import OrderedDict

import numpy as np
import torch

from models import vgg_16_bn, ResNet18_widar, ResNet18_cifar, MobileNetV2


def Aggregation(w, lens):
    w_avg = None
    total_count = sum(lens)

    for i in range(0, len(w)):
        if i == 0:
            w_avg = copy.deepcopy(w[0])
            for k in w_avg.keys():
                w_avg[k] = w[i][k] * lens[i]
        else:
            for k in w_avg.keys():
                w_avg[k] += w[i][k] * lens[i]

    for k in w_avg.keys():
        w_avg[k] = torch.div(w_avg[k], total_count)

    return w_avg


def split_model(global_param, slim_param):
    param = copy.deepcopy(slim_param)
    for k, v in param.items():  # 遍历所有层，每一层遍历
        if v.dim() > 1:
            d1 = v.shape[0]
            d2 = v.shape[1]
            param[k] = global_param[k][:d1, :d2]
        elif v.dim() == 1:
            d1 = v.shape[0]
            param[k] = global_param[k][:d1]
        else:
            param[k] = global_param[k]
    return param


def Aggregation_AdaptiveFL(w, lens, global_model_param):
    w_avg = copy.deepcopy(global_model_param)  # largest model
    count = OrderedDict()
    for k, v in w_avg.items():  # 遍历所有层，每一层遍历
        parameter_type = k.split('.')[-1]

        count[k] = v.new_zeros(v.size(), dtype=torch.float32)
        tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
        for m in range(len(w)):  # 遍历所有用户
            if parameter_type == 'weight':
                if v.dim() > 1:  # 卷积  和 线性层
                    d1 = w[m][k].shape[0]
                    d2 = w[m][k].shape[1]
                    tmp_v[:d1, :d2] += w[m][k] * lens[m]  # 第m个客户端的 k 层参数
                    count[k][:d1, :d2] += lens[m]
                else:  # BN层
                    d1 = w[m][k].shape[0]
                    tmp_v[:d1] += w[m][k] * lens[m]
                    count[k][:d1] += lens[m]
            else:
                d1 = w[m][k].shape[0]
                tmp_v[:d1] += w[m][k] * lens[m]
                count[k][:d1] += lens[m]

        tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
        tmp_v[count[k] == 0] = global_model_param[k][count[k] == 0]
        w_avg[k] = tmp_v

    return w_avg


def Aggregation_ScaleFL(w, lens, grad_info, global_model_param):
    w_avg = copy.deepcopy(global_model_param)  # largest model
    count = OrderedDict()
    for idx, (k, v) in enumerate(w_avg.items()):  # 遍历所有层，每一层遍历
        parameter_type = k.split('.')[-1]

        count[k] = v.new_zeros(v.size(), dtype=torch.float32)
        tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
        for m in range(len(w)):  # 遍历所有用户
            if grad_info[m][idx]:
                if parameter_type == 'weight':
                    if v.dim() > 1:  # 卷积  和 线性层
                        d1 = w[m][k].shape[0]
                        d2 = w[m][k].shape[1]
                        tmp_v[:d1, :d2] += w[m][k] * lens[m]  # 第m个客户端的 k 层参数
                        count[k][:d1, :d2] += lens[m]
                    else:  # BN层
                        d1 = w[m][k].shape[0]
                        tmp_v[:d1] += w[m][k] * lens[m]
                        count[k][:d1] += lens[m]
                else:
                    d1 = w[m][k].shape[0]
                    tmp_v[:d1] += w[m][k] * lens[m]
                    count[k][:d1] += lens[m]


        tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
        tmp_v[count[k] == 0] = global_model_param[k][count[k] == 0]
        w_avg[k] = tmp_v

    return w_avg

def Aggregation_LiquidFL(w, lens, global_model_param, alpha=0.5,
                         return_stats=False):
    """Coverage-aware delta aggregation for persistent heterogeneous models.

    For every global coordinate, first compute the conditional mean update
    among clients that actually contain that coordinate.  Then attenuate it
    by ``coverage ** alpha``, where coverage is the participating sample
    weight divided by the total sample weight in this round.

    ``alpha=0`` recovers AdaptiveFL's conditional coordinate average;
    ``alpha=1`` recovers a zero-padded FedAvg delta.  The default 0.5 keeps
    learning in outer channels while preventing a few permanently strong,
    non-IID clients from overwriting them at full strength.
    """
    if not w or len(w) != len(lens):
        raise ValueError('w and lens must be non-empty and have equal length')
    if any(float(n) < 0.0 for n in lens):
        raise ValueError('client sample weights must be nonnegative')
    total_weight = float(sum(lens))
    if total_weight <= 0.0:
        raise ValueError('total client sample weight must be positive')
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('LiquidFL aggregation alpha must be in [0, 1]')

    result = copy.deepcopy(global_model_param)
    coverage_sum = 0.0
    scale_sum = 0.0
    coordinate_count = 0
    zero_count = 0
    full_count = 0

    for key, global_value in global_model_param.items():
        if not torch.is_floating_point(global_value):
            # Integer bookkeeping buffers must never receive fractional
            # deltas. LiquidFL currently disables BN running statistics, but
            # preserving them makes this routine robust to future backbones.
            result[key] = global_value.detach().clone()
            continue

        delta_sum = torch.zeros_like(global_value)
        active_weight = torch.zeros_like(global_value, dtype=torch.float32)
        for local_state, sample_weight in zip(w, lens):
            if key not in local_state:
                raise KeyError(f'local model is missing state key: {key}')
            local_value = local_state[key]
            if local_value.dim() != global_value.dim():
                raise ValueError(
                    f'incompatible rank for {key}: local={local_value.shape}, '
                    f'global={global_value.shape}')
            if any(local_size > global_size for local_size, global_size in zip(
                    local_value.shape, global_value.shape)):
                raise ValueError(
                    f'local tensor is larger than global tensor for {key}: '
                    f'local={local_value.shape}, global={global_value.shape}')

            weight = float(sample_weight)
            if weight == 0.0:
                continue
            if local_value.dim() == 0:
                delta_sum.add_(
                    (local_value.to(delta_sum.dtype) - global_value) * weight)
                active_weight.add_(weight)
            else:
                slices = tuple(slice(0, size) for size in local_value.shape)
                reference = global_value[slices]
                delta_sum[slices].add_(
                    (local_value.to(delta_sum.dtype) - reference) * weight)
                active_weight[slices].add_(weight)

        mask = active_weight > 0
        coverage = active_weight / total_weight
        scale = coverage.pow(alpha)
        updated = global_value.detach().clone()
        updated[mask] = (
            global_value[mask]
            + scale[mask] * delta_sum[mask] / active_weight[mask]
        )
        result[key] = updated

        numel = global_value.numel()
        coordinate_count += numel
        coverage_sum += float(coverage.sum().item())
        scale_sum += float(scale[mask].sum().item())
        zero_count += int((~mask).sum().item())
        full_count += int((coverage >= 1.0 - 1e-7).sum().item())

    stats = {
        'alpha': alpha,
        'mean_coordinate_coverage': (
            coverage_sum / max(coordinate_count, 1)),
        'mean_active_scale': scale_sum / max(
            coordinate_count - zero_count, 1),
        'zero_coverage_fraction': zero_count / max(coordinate_count, 1),
        'full_coverage_fraction': full_count / max(coordinate_count, 1),
    }
    return (result, stats) if return_stats else result

def get_model_list(args):
    model_rate = args.width_ration
    depth_list = args.depth_saved

    net_glob_list = []
    net_slim_info = []
    for i in model_rate:
        for depth in depth_list:
            if i < 1.0 and depth == 2:
                continue
            
            if args.model == 'vgg':
                net = vgg_16_bn(num_classes=args.num_classes, track_running_stats=False, num_channels=args.num_channels, slim_idx=depth, scale=i)
            elif args.model == 'resnet':
                if args.dataset == 'widar':
                    net = ResNet18_widar(num_classes=args.num_classes, track_running_stats=False, slim_idx=depth, scale=i)
                else:
                    net = ResNet18_cifar(num_channels=args.num_channels, num_classes=args.num_classes, track_running_stats=False, slim_idx=depth, scale=i)

            elif args.model == 'mobilenet':
                net = MobileNetV2(channels= args.num_channels, num_classes=args.num_classes, trs=False, slim_idx=depth, scale=i)

            total = sum([param.nelement() for param in net.parameters()])
            net.to(args.device)
            net.train()
            print("==" * 50)
            print('【model config】  model_name:{}, width:{} , depth:{}, param:{}MB'.format(args.model, i, depth, total * 4 / 1e6))
            # print(net)
            net_glob_list.append(net)
            net_slim_info.append((i, depth, total / 1e6))  # 宽度 深度 参数量

            if i == 1.0:
                break
    return net_glob_list, net_slim_info


def _profile_resource_group(model_type, net_glob_list_len):
    """Return the minimum client-resource tier required by a profile."""
    model_type = int(model_type)
    if net_glob_list_len == 7:
        group = model_type // 3
    elif net_glob_list_len == 5:
        # Five-profile family: (P1, P2), (P4, P5), P6.
        group = 0 if model_type < 2 else (1 if model_type < 4 else 2)
    elif net_glob_list_len == 3:
        group = model_type
    else:
        raise ValueError(
            'Unsupported profile count {} for rule-based client selection'.format(
                net_glob_list_len))

    if model_type < 0 or model_type >= net_glob_list_len or group not in (0, 1, 2):
        raise ValueError(
            'Invalid profile index {} for {} profiles'.format(
                model_type, net_glob_list_len))
    return group


def select_clients(args, ration_users, net_glob_list_len):
    my_list = list(map(float, args.client_hetero_ration.split(':')))
    if len(my_list) != 3 or any(x <= 0 for x in my_list):
        raise ValueError('client_hetero_ration must contain three positive tiers')
    hetero_proportion = [x / sum(my_list) for x in my_list]

    if args.client_chosen_mode == 'random':
        if len(ration_users) > args.num_users:
            raise ValueError('Cannot sample more unique clients than num_users')
        return random.sample(range(args.num_users), len(ration_users))

    if args.client_chosen_mode not in ('available', 'fit'):
        raise ValueError(
            'select_clients does not support mode {}'.format(
                args.client_chosen_mode))

    # Existing client ids are ordered weak -> medium -> strong.
    boundaries = [
        0,
        int(args.num_users * hetero_proportion[0]),
        int(args.num_users * sum(hetero_proportion[:2])),
        args.num_users,
    ]
    idx_users = []
    for model_type in ration_users:
        group = _profile_resource_group(model_type, net_glob_list_len)
        low = boundaries[group]
        high = (args.num_users - 1
                if args.client_chosen_mode == 'available'
                else boundaries[group + 1] - 1)
        if low > high:
            raise ValueError(
                'Client tier {} is empty under client_hetero_ration={}'.format(
                    group, args.client_hetero_ration))
        idx_users.append(random.randint(low, high))

    # Preserve one-to-one, ordered profile/client pairing.
    assert len(idx_users) == len(ration_users)
    return idx_users
