#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
from tqdm import tqdm

from Algorithm.Training_Decoupled import Decoupled
from Algorithm.Training_ScaleFL import ScaleFL
import torch
# matplotlib.use('Agg')
import copy

from utils.Clients import Clients
from utils.options import args_parser
from models import *
from utils.get_datatset_new import get_dataset
from utils.utils import save_result
from utils.set_seed import set_random_seed
from Algorithm.Training_AdaptiveFL import AdaptiveFL
from Algorithm.Training_HeteroFL import HeteroFL

import sys
import logging
from datetime import datetime
# =============================================================================
# Tee logger: print to terminal + file
# =============================================================================
class TeeLogger:
    """Write stdout to both console and a file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def f2s(x: float) -> str:
    # 0.6 -> "0p6"  (避免文件名里出现多个点或奇怪格式)
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")



import sys

class TeeOutput:
    """同时写终端和文件"""
    def __init__(self, file_path):
        self.terminal = sys.__stdout__  # 用原始stdout，避免嵌套重定向
        self.file = open(file_path, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


if __name__ == '__main__':
    run_dir = 'result_ori_AdapterFL_5'
    os.makedirs(run_dir, exist_ok=True)

    algorithms = ['AdaptiveFL']
    datasets   = ['cifar10']
    settings   = [
        # {'iid': 0, 'data_beta': 0.6},
        {'iid': 0, 'data_beta': 0.3},
        # {'iid': 1, 'data_beta': 1.0},
    ]

    for algorithm in algorithms:
        for dataset in datasets:
            for setting in settings:
                # try:
                    stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
                    iid_flag  = setting['iid']
                    data_beta = setting['data_beta']
                    tag       = 'iid' if iid_flag else f"beta{data_beta}"

                    log_path = os.path.join(
                        run_dir,
                        f"{tag}_{dataset}_{algorithm}_resnet_{stamp}.log")

                    # ── 重定向 stdout 到 tee ───────────────────
                    tee = TeeOutput(log_path)
                    sys.stdout = tee

                    print(f"\n{'='*60}")
                    print(f"algorithm={algorithm} dataset={dataset} {tag}")
                    print(f"{'='*60}")

                    args = args_parser()
                    args.device = torch.device(
                        'cuda:{}'.format(args.gpu)
                        if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

                    args.model     = 'resnet'
                    args.dataset   = dataset
                    args.iid       = iid_flag
                    args.data_beta = data_beta
                    args.algorithm = algorithm
                    args.num_classes = 10 if dataset == 'cifar10' else 100

                    set_random_seed(args.seed)

                    dataset_train, dataset_test, dict_users = get_dataset(args)

                    if args.dataset == 'widar':
                        net_glob = ResNet18_widar(num_classes=args.num_classes)
                    else:
                        net_glob = ResNet18_cifar(
                            num_channels=args.num_channels,
                            num_classes=args.num_classes)

                    if algorithm == 'AdaptiveFL':
                        args.depth_saved  = [2, 3, 4]
                        args.width_ration = [0.4, 0.66, 1.0]
                        AdaptiveFL(args, dataset_train, dataset_test, dict_users)
                    elif algorithm == 'HeteroFL':
                        args.depth_saved  = [0]
                        args.width_ration = [0.5, 0.71, 1.0]
                        HeteroFL(args, dataset_train, dataset_test, dict_users)
                    elif algorithm == 'ScaleFL':
                        args.depth_saved  = [2, 3, 4]
                        args.width_ration = [0.4, 0.66, 1.0]
                        ScaleFL(args, dataset_train, dataset_test, dict_users)
                    elif algorithm == 'Decoupled':
                        args.depth_saved  = [8]
                        args.width_ration = [0.4, 0.66, 1.0]
                        Decoupled(args, dataset_train, dataset_test, dict_users)

                    print(f"✓ 完成: {algorithm} {dataset} {tag}")

                # except Exception as e:
                #     print(f"✗ 失败: {algorithm} {dataset} {tag} | {e}")
                #     import traceback; traceback.print_exc()

                # finally:
                #     # ── 每组结束后恢复 stdout ──────────────────
                #     sys.stdout = sys.__stdout__
                #     if tee is not None:
                #         tee.close()

    print("\n所有 baseline 实验完成")