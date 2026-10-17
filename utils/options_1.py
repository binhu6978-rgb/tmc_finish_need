#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=1000, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="paper-spec constant learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="paper-spec weight decay")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument("--algorithm", type=str, default="AdaptiveFL")

    # model arguments
    parser.add_argument('--model', type=str, default='resnet', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--use_project_head', type=int, default=0)
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar100', help="name of dataset")
    parser.add_argument('--generate_data', type=int, default=1, help="whether generate new dataset")
    parser.add_argument('--iid', type=int, default=0, help='whether i.i.d or not')
    parser.add_argument('--noniid_case', type=int, default=4, help="non i.i.d case (1, 2, 3, 4)")
    parser.add_argument('--data_beta', type=float, default=0.3,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--num_classes', type=int, default=100, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--partition_seed', type=int, default=None,
                        help='independent partition seed; defaults to --seed')
    parser.add_argument('--measurement_head_seed', type=int, default=None,
                        help='isolated seed for the frozen SemConsFL measurement head')
    parser.add_argument('--diagnostic_enabled', action='store_true',
                        help='enable read-only EXP-1 diagnostics')
    parser.add_argument('--diagnostic_fraction', type=float, default=0.05)
    parser.add_argument('--diagnostic_split_seed', type=int, default=1)
    parser.add_argument('--diagnostic_every', type=int, default=50)
    parser.add_argument('--diagnostic_batch_size', type=int, default=128)
    parser.add_argument('--revision_output_dir', type=str, default='revision/raw')
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--gate_mode', type=str, default='semcons', choices=['semcons'])
    parser.add_argument('--n_cons', type=int, default=3,
                        help='explicit provisional canonical default')
    # FedProx
    parser.add_argument('--prox_alpha', type=float, default=0.01, help='The hyper parameter for the FedProx')
    # SCAFFOLD
    parser.add_argument('--lr_g', type=float, default=0.1, help="global learning rate for SCAFFOLD")
    # Moon
    parser.add_argument('--contrastive_alpha', type=float, default=5, help='The hyper parameter for the Moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--model_buffer_size', type=int, default=1,
                        help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    # FedGKD
    parser.add_argument('--ensemble_alpha', type=float, default=0.2, help='The hyper parameter for the FedGKD')
    # FedDC
    parser.add_argument('--sim_type', type=str, default='L1', help='Cluster Sampling: cosine or L1 or L2')
    # FedDC
    parser.add_argument('--alpha_coef', type=float, default=1e-2, help='FedDC')
    # FedMLB
    parser.add_argument("--temp", default=1, type=float, metavar="N", help="temperature")
    parser.add_argument("--lambda1", default=1, type=float, metavar="N", help="Weight for CE loss of main pathway")
    parser.add_argument("--lambda2", default=1, type=float, metavar="N", help="Weight for CE loss of hybrid pathways")
    parser.add_argument("--lambda3", default=1, type=float, metavar="N", help="Weight for KD loss of hybrid pathways")


    # FedSlim
    parser.add_argument("--client_chosen_mode", default='random', type=str,
                        help="available / fit / random / RL")
    parser.add_argument("--depth_saved", default=[0], type=int, nargs='*',
                        help="SemConsFL uses width-only nested submodels")
    parser.add_argument("--width_ration", default=[0.5, 0.75, 1.0], type=float, nargs='*',
                        help="paper-spec SemConsFL widths")
    parser.add_argument("--client_hetero_ration", default='4:3:3', type=str, help="the info of model ration and model type.")


    args = parser.parse_args()
    return args
'''
# vgg 采用4-6-8 resnet采用2-3-4 其中resnet是用block为最小单位
FedScale:  depth[4, 6, 8] width[0.4, 0.66, 1.0]
HeteroFL: depth[0] width[0.5, 0.71, 1.0]
Decoupled:depth[8] width[0.4, 0.66, 1.0]
'''
