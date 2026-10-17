#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command-line options for the fixed client-submodel training path."""

import argparse


def args_parser():
    parser = argparse.ArgumentParser(
        description='Federated learning with fixed client submodel shapes')

    training = parser.add_argument_group('federated training')
    training.add_argument('--epochs', type=int, default=1000)
    training.add_argument('--num_users', type=int, default=100)
    training.add_argument('--frac', type=float, default=0.1)
    training.add_argument('--local_ep', type=int, default=5)
    training.add_argument('--local_bs', type=int, default=50)
    training.add_argument('--bs', type=int, default=128)
    training.add_argument(
        '--optimizer', choices=('sgd', 'adam'), default='sgd')
    training.add_argument('--lr', type=float, default=0.01)
    training.add_argument('--lr_decay', type=float, default=0.998)
    training.add_argument('--momentum', type=float, default=0.5)
    training.add_argument('--weight_decay', type=float, default=1e-4)

    distillation = parser.add_argument_group('nested self-distillation')
    distillation.add_argument('--T', type=float, default=3.0)
    distillation.add_argument('--gamma', type=float, default=10.0)
    distillation.add_argument('--kd_decay_start', type=int, default=-1)
    distillation.add_argument('--kd_decay_end', type=int, default=-1)
    distillation.add_argument('--kd_final_ratio', type=float, default=0.1)

    data = parser.add_argument_group('data')
    data.add_argument(
        '--dataset', choices=('cifar10', 'cifar100', 'tiny_imagenet'),
        default='cifar10')
    data.add_argument('--data_dir', type=str, default='./data')
    data.add_argument('--generate_data', type=int, choices=(0, 1), default=1)
    data.add_argument('--iid', type=int, choices=(0, 1), default=0)
    data.add_argument('--data_beta', type=float, default=0.3)
    data.add_argument('--noniid_case', type=int, default=4)
    data.add_argument('--min_size', type=int, default=10)
    data.add_argument('--min_classes', type=int, default=0)
    data.add_argument('--max_retry', type=int, default=500)

    model = parser.add_argument_group('model and resources')
    model.add_argument(
        '--model', choices=('resnet', 'mobilenet'), default='resnet')
    model.add_argument('--num_classes', type=int, default=10)
    model.add_argument('--num_channels', type=int, default=3)
    model.add_argument('--depth_saved', type=int, nargs='*', default=[2, 3, 4])
    model.add_argument(
        '--width_ration', type=float, nargs='*', default=[0.4, 0.66, 1.0])
    model.add_argument('--client_hetero_ration', type=str, default='4:3:3')

    fixed = parser.add_argument_group('fixed submodel assignment')
    fixed.add_argument(
        '--fixed_profile_policy', default='tier_balanced',
        choices=('tier_balanced', 'tier_max'),
        help='one-time assignment inside each client resource tier')
    fixed.add_argument(
        '--fixed_assignment_seed', type=int, default=-1,
        help='-1 reuses the experiment seed')
    fixed.add_argument('--fixed_eval_interval', type=int, default=1)

    runtime = parser.add_argument_group('runtime')
    runtime.add_argument('--gpu', type=int, default=0)
    runtime.add_argument('--seed', type=int, default=42)
    runtime.add_argument('--verbose', action='store_true')
    runtime.add_argument('--algorithm', type=str, default='StaticSubmodelFL')
    runtime.add_argument(
        '--macs_coeff', type=float, default=3.0,
        help='student train MAC multiplier for forward/backward/optimizer')
    runtime.add_argument('--comm_bytes_per_param', type=float, default=4.0)
    return parser.parse_args()


__all__ = ['args_parser']
