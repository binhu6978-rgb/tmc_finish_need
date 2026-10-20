#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import time
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from torch import nn
import copy
import numpy as np
from tqdm import tqdm

from models.Fed import get_model_list, Aggregation_AdaptiveFL, split_model, select_clients
from models.vgg import vgg_16_bn
from models.resnet import ResNet18_cifar
from utils.HeteroClients import HeteroClients
from utils.utils import save_result, my_save_result, get_final_acc
from models.test import test_img, test
from models.Update import DatasetSplit, LocalUpdate_AdaptiveFL
from optimizer.Adabelief import AdaBelief


def AdaptiveFL(args, dataset_train, dataset_test, dict_users):
    net_glob_list, net_slim_info = get_model_list(args)

    # training
    total_time = 0
    time_list = []
    acc_list = [[] for _ in net_glob_list]
    clients = HeteroClients(args, net_slim_info)

    # Explicit opt-in only. Without --method, the original path stays intact.
    method_run = None
    isolated_eval = getattr(args, 'eval_profiles', 'all') == 'full' or getattr(args, 'isolate_eval_rng', False)
    if isolated_eval and not getattr(args, 'method', None):
        args.method = 'baseline'
    if getattr(args, 'method', None):
        from adapterfl_methods import RunLog
        method_run = RunLog(args, net_glob_list)
        total_time = method_run.total_time

    # 开始训练
    for iter in tqdm(range(method_run.next_round if method_run else 0, args.epochs)):  # tqdm 进度条库

        round_begin = time.perf_counter() if method_run else None
        scope = method_run.hooks if method_run and getattr(method_run.hooks,'is_scope',False) else None
        if scope:scope.start_round(iter)

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []
        dispatched_snapshots = []
        local_diagnostics = []

        m = max(int(args.frac * args.num_users), 1)

        # 用户选择
        if args.client_chosen_mode == 'RL':  # RL-Selection
            # ration_users = np.random.choice([2, 5, 6], m)  # 模型选择
            # print('RL')
            ration_users = np.random.choice(range(len(net_glob_list)), m)  # 模型选择
            idx_users = clients.select_clients(ration_users)  # 基于强化学习的客户端选择
        elif args.client_chosen_mode == 'greedy':  # greedy 选择
            ration_users = np.random.choice([len(net_glob_list)-1], m)  # 模型选择
            idx_users = random.sample(range(args.num_users), len(ration_users))  # 基于强化学习的客户端选择
        else:
            ration_users = np.random.choice(range(len(net_glob_list)), m)  # 模型选择
            idx_users = select_clients(args, ration_users, len(net_glob_list))  # 基于规则的客户端选择

        feedback_list = []
        max_time = 0
        for id, idx in enumerate(idx_users):
            begin = time.time()
            local = LocalUpdate_AdaptiveFL(args=args, dataset=dataset_train, idxs=dict_users[idx])
            if args.client_chosen_mode == 'RL' or args.client_chosen_mode == 'random' or args.client_chosen_mode == 'greedy':  # 在客户端本地做自适应裁剪
                feedback_model_idx = clients.train(idx, ration_users[id])
            elif args.client_chosen_mode == 'available' or args.client_chosen_mode == 'fit':
                feedback_model_idx = ration_users[id]  # 这里是基于规则的，不会出错，分发的是什么模型，返回的就是什么模型
            feedback_list.append(feedback_model_idx)

            if method_run:
                method_run.hooks.configure_local(local)
                if args.method in ('m2', 'm5', 'c1', 'c2') or getattr(method_run.hooks,'is_tonight',False) or getattr(method_run.hooks,'is_local12',False) or (getattr(method_run.hooks,'is_native_groups',False) and method_run.hooks.needs_dispatch) or (scope and scope.needs_dispatch):
                    # These are the actual per-slot pre-training weights, including
                    # independent first-round submodel initialization.
                    dispatched_snapshots.append(copy.deepcopy(net_glob_list[feedback_model_idx].state_dict()))

            if scope:
                local_net = copy.deepcopy(net_glob_list[feedback_model_idx]).to(args.device)
                before = scope.before_slot(local_net,dataset_train,dict_users[idx],int(idx))
                native_begin = time.perf_counter()
                w = local.train(round=iter, net=local_net)
                native_seconds = time.perf_counter()-native_begin
                local.adapterfl_metrics=scope.finish_slot(local_net,dataset_train,before,int(idx),int(feedback_model_idx))
                local.adapterfl_metrics['native_local_train_seconds']=native_seconds
            else:
                w = local.train(round=iter, net=copy.deepcopy(net_glob_list[feedback_model_idx]).to(args.device))

            if method_run:
                local_diagnostics.append(getattr(local, 'adapterfl_metrics', {}))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
            time_epoch = time.time() - begin
            max_time = max(max_time, time_epoch)

        total_time += max_time

        time_list.append(total_time)
        print(f"this epoch cost time:{max_time}")

        print(f"this epoch choose: {idx_users}")  # 这一轮选择的用户下标
        print(f"this epoch dispatch models: {ration_users}")  # 每个用户对应的模型的比例, 初始分发
        print(f"this epoch received models: {feedback_list}")  # 每个用户对应的模型的比例, 初始分发
        print(f"hetero_proportion: \t{args.client_hetero_ration}")
        # 需要print 每个客户端的计算资源

        server_begin = time.perf_counter() if method_run else None
        if method_run:
            if getattr(method_run.hooks,'is_tonight',False) or getattr(method_run.hooks,'is_local12',False) or getattr(method_run.hooks,'is_native_groups',False):
                method_run.hooks.set_slots(zip(idx_users,feedback_list))
            w_glob_param = method_run.hooks.aggregate(w_locals, lens, net_glob_list[-1].state_dict(), dispatched_snapshots)
        else:
            w_glob_param = Aggregation_AdaptiveFL(w_locals, lens, net_glob_list[-1].state_dict())

        server_seconds = time.perf_counter() - server_begin if method_run else None
        evaluations = []
        evaluation_seconds = 0.

        if isolated_eval:
            # Synchronize all seven, even when only Full is evaluated.
            for net in net_glob_list:
                net.load_state_dict(split_model(w_glob_param, net.state_dict()))
            from adapterfl_methods import isolated_evaluation_rng
            evaluations = {}
            indices = [len(net_glob_list)-1] if args.eval_profiles == 'full' else range(len(net_glob_list))
            evaluation_begin = time.perf_counter()
            with isolated_evaluation_rng():
                for idx in indices:
                    accuracy, ce = test_img(net_glob_list[idx], dataset_test, args)
                    print(net_slim_info[idx])
                    print("Testing accuracy: {:.2f}".format(accuracy))
                    acc_list[idx].append(accuracy)
                    evaluations[idx] = (accuracy, ce)
            evaluation_seconds = time.perf_counter() - evaluation_begin
        else:
            # Frozen legacy all-profile evaluation order and RNG consumption.
            for idx, net in enumerate(net_glob_list):
                net.load_state_dict(split_model(w_glob_param, net.state_dict()))
                print(net_slim_info[idx])
                if method_run:
                    evaluation_begin = time.perf_counter()
                    accuracy, ce = test_img(net, dataset_test, args)
                    evaluation_seconds += time.perf_counter() - evaluation_begin
                    print("Testing accuracy: {:.2f}".format(accuracy))
                    acc_list[idx].append(accuracy)
                    evaluations.append((accuracy, ce))
                else:
                    acc_list[idx].append(test(net, dataset_test, args))

        if method_run:
            method_run.record(iter, net_glob_list, evaluations, max_time, total_time,
                              time.perf_counter() - round_begin, server_seconds, evaluation_seconds,
                              [(int(c), int(p)) for c, p in zip(idx_users, feedback_list)], local_diagnostics)

    if method_run:
        method_run.finish()
        return

    save_result(time_list, 'test_time', args)
    for id, acc in enumerate(acc_list):
        file = my_save_result(acc, str(net_slim_info[id]), 'acc', args)
        # save_result(acc, 'accuracy', args)
    get_final_acc(file)


'''
                    
            for id, p in enumerate(model_proportion):
                if id == 0:  # small model
                    idx_users = np.random.choice(range(args.num_users), int(m * p), replace=False)
                    ration_users = [random.randint(0, 2) for _ in range(int(m * p))]
                elif id == 1:  # medium model
                    idx_users = np.hstack((idx_users,
                                           np.random.choice(
                                               [i for i in
                                                range(int(args.num_users * sum(hetero_proportion[:id])), args.num_users)  # 做个sum
                                                if i not in idx_users],
                                               int(m * p), replace=False)))
                    ration_users.extend([random.randint(3, 5) for _ in range(int(m * p))])
                elif id == 2:  # large model
                    idx_users = np.hstack((idx_users,
                                           np.random.choice(
                                               [i for i in
                                                range(int(args.num_users * sum(hetero_proportion[:id])), args.num_users)  # 做个sum
                                                if i not in idx_users],
                                               int(m * p), replace=False)))
                    ration_users.extend([6 for _ in range(int(m * p))])
'''
