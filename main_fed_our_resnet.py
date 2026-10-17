#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fixed-submodel ResNet18 experiment on CIFAR-100 Dirichlet alpha=0.3."""

import os
import sys
import time
import traceback
from datetime import datetime

import torch

from Algorithm.Training_StaticSubmodelFL import StaticSubmodelFL
from utils.get_datatset_new import get_dataset
from utils.options_static import args_parser
from utils.set_seed import set_random_seed


class TeeStream:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()


class ExperimentLogger:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.stdout, self.stderr = sys.stdout, sys.stderr
        self.stream = open(self.path, 'w', encoding='utf-8', buffering=1)
        sys.stdout = TeeStream(self.stdout, self.stream)
        sys.stderr = TeeStream(self.stderr, self.stream)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout, sys.stderr = self.stdout, self.stderr
            self.stream.close()


def main():
    args = args_parser()
    # This dedicated launcher intentionally fixes the experiment identity.
    # Other CLI options (epochs, seed, optimizer, client fraction, etc.)
    # remain configurable.
    args.model = 'resnet'
    args.dataset = 'cifar100'
    args.iid = 0
    args.data_beta = 0.3
    args.num_classes = 100
    args.num_channels = 3
    args.device = torch.device(
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.algorithm = 'StaticSubmodelFL'

    run_dir = 'result_static_submodel'
    os.makedirs(run_dir, exist_ok=True)
    args.output_dir = os.path.join(run_dir, 'output')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.run_stamp = stamp
    setting = 'iid' if bool(args.iid) else 'beta{}'.format(args.data_beta)
    log_path = os.path.join(
        run_dir, '{}_{}_{}_{}_{}.log'.format(
            setting, args.dataset, args.algorithm, args.model, stamp))

    print('Log -> {}'.format(os.path.abspath(log_path)))
    with ExperimentLogger(log_path):
        start = time.time()
        try:
            print('=' * 72)
            print('Static client-submodel federated experiment')
            for key, value in sorted(vars(args).items()):
                print('{:28s}: {}'.format(key, value))
            print('=' * 72)
            set_random_seed(args.seed)
            train_set, test_set, users = get_dataset(args)
            StaticSubmodelFL(args, train_set, test_set, users)
            print('elapsed_seconds: {:.2f}'.format(time.time() - start))
        except Exception:
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
