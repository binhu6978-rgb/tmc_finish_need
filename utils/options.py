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
    parser.add_argument('--lr_decay', type=float, default=0.998, help="learning rate decay")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight_decay (default: 1e-4)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument("--algorithm", type=str, default="AdaptiveFL")
    parser.add_argument('--lambda_cost',  type=float, default=0.03)  # 原 0.15
    parser.add_argument('--delta_r',      type=float, default=0.03)  # 原 0.05

    parser.add_argument('--pool_warmup',  type=int,   default=100)
    parser.add_argument('--proxy_window', type=int,   default=20)
    parser.add_argument('--pool_update_interval', type=int,   default=50)
    parser.add_argument('--pool_delta',           type=float, default=0.02)
    parser.add_argument('--pool_threshold',       type=float, default=0.70)
    parser.add_argument('--r1_min',               type=float, default=0.25)
    parser.add_argument('--r2_min',               type=float, default=0.5)
    parser.add_argument('--pool_use_test_proxy', type=int, default=0)
    # model arguments

    parser.add_argument('--use_lpc',        type=int,   default=1)
    parser.add_argument('--lpc_gamma',      type=float, default=0.25)
    parser.add_argument('--lpc_alpha',      type=float, default=0.3)
    parser.add_argument('--lpc_lr',         type=float, default=1e-3)
    parser.add_argument('--lpc_g_init',     type=float, default=0.0)
    parser.add_argument('--lpc_tolerance',  type=float, default=0.005)
    parser.add_argument('--lpc_imit_steps', type=int,   default=10)

    # LiquidFL-5P uses five nested APOZ profiles matched to AdaptiveFL's
    # realized parameter budgets. Persistent capacity and liquid memories are
    # allocated lazily for promoted clients from every resource tier. F4 is
    # immutable, but strong clients can personalize sampled F0--F3 profiles.
    parser.add_argument('--liquid_warmup', type=int, default=20)
    parser.add_argument('--liquid_ctrl_interval', type=int, default=1,
                        help='control-round period (rounds, or participations '
                             'if --liquid_ctrl_by_participation=1)')
    parser.add_argument('--liquid_ctrl_by_participation', type=int, default=0,
                        help='1: trigger control by the client\'s own '
                             'participation count instead of global round')
    parser.add_argument('--liquid_min_participations', type=int, default=2,
                        help='minimum completed client participations before '
                             'Need-Gate promotion is allowed')
    parser.add_argument('--liquid_controller_lr', type=float, default=1e-3)
    parser.add_argument(
        '--liquid_controller_variant', type=str, default='full',
        choices=('performance', 'full', 'random', 'gru', 'reset'),
        help='full is the performance-validated default controller; '
             'performance keeps the experimental time-aware variant; '
             'remaining choices are controller ablations')
    parser.add_argument(
        '--liquid_strong_mode', type=str, default='personalize',
        choices=('personalize', 'full_route', 'static'),
        help='strong-tier behavior; the final method uses personalize')
    parser.add_argument('--liquid_delta', type=float, default=0.05,
                        help='max per-control-round capacity increment (L-inf)')
    parser.add_argument('--liquid_m_min', type=float, default=0.2)
    parser.add_argument('--liquid_tau', type=float, default=1.0,
                        help='soft prefix-gate temperature in channel-index '
                             'units for q feedback')
    parser.add_argument('--liquid_feedback_bs', type=int, default=32)
    parser.add_argument('--liquid_hidden_dim', type=int, default=32)
    # ---- Need Gate ----
    parser.add_argument('--liquid_need_patience', type=int, default=2,
                        help='consecutive control participations satisfying '
                             'raw_need before the gate opens')
    parser.add_argument('--liquid_reuse_threshold_scale', type=float,
                        default=0.5,
                        help='difficulty/stagnation threshold multiplier for '
                             'clients that already own a personalized subnet')
    parser.add_argument('--liquid_reuse_hold', type=int, default=3,
                        help='number of consecutive personalized uses after '
                             'an active client passes the relaxed reuse gate')
    parser.add_argument('--liquid_reuse_explore', type=float, default=0.10,
                        help='audit probability for an active personalized '
                             'client when its relaxed reuse gate is closed')
    parser.add_argument('--liquid_self_difficulty_threshold', type=float,
                        default=0.05, help='z_self (relative worsening) gate')
    parser.add_argument('--liquid_pool_difficulty_threshold', type=float,
                        default=1.0, help='z_pool (standard score) gate')
    parser.add_argument('--liquid_drop_threshold', type=float, default=0.02,
                        help='relative-drop threshold for the stagnation score')
    parser.add_argument('--liquid_stagnation_threshold', type=float,
                        default=0.30, help='stagnation_ema gate in [0, 1]')
    parser.add_argument('--liquid_stagnation_ema_decay', type=float, default=0.8)
    parser.add_argument('--liquid_loss_floor', type=float, default=0.0,
                        help='>0 enables absolute-loss difficulty protection')
    parser.add_argument('--liquid_ema_warmup', type=int, default=2,
                        help='updates after re-anchor before z_self is trusted')
    parser.add_argument('--liquid_probe_ema_decay', type=float, default=0.9,
                        help='EMA decay for the per-client self probe loss')
    # ---- pooled baseline tracker (coarse capacity bucket) ----
    parser.add_argument('--liquid_bucket_step', type=float, default=0.10,
                        help='capacity quantization step for the pooled bucket')
    parser.add_argument('--liquid_sig_min_count', type=int, default=5,
                        help='min bucket count before z_pool is used')
    parser.add_argument('--liquid_pool_std_floor', type=float, default=0.02,
                        help='std floor for z_pool numerical stability')
    parser.add_argument('--liquid_baseline_ema_decay', type=float, default=0.9)
    # ---- expert update ----
    parser.add_argument('--liquid_q_clip', type=float, default=1.0,
                        help='L2-norm clipping radius for capacity sensitivity')
    parser.add_argument('--liquid_grad_clip', type=float, default=5.0)
    parser.add_argument('--liquid_move_reg', type=float, default=1e-3)
    parser.add_argument(
        '--liquid_action_floor', type=float, default=0.1,
        help='minimum non-zero normalized action magnitude before multiplying '
             'by liquid_delta; prevents learned changes from remaining below '
             'hard-channel quantization resolution')
    parser.add_argument(
        '--liquid_gap_scale', type=float, default=10.0,
        help='round-gap normalization used by time-aware liquid memories')
    parser.add_argument('--liquid_qrms_ema', type=float, default=0.9,
                        help='>0 stabilizes q scale by a running RMS (else use '
                             'raw magnitude)')
    parser.add_argument('--liquid_qrms_floor', type=float, default=1e-3,
                        help='minimum q scale to avoid amplifying tiny noise')
    parser.add_argument('--liquid_growth_phase_ratio', type=float, default=0.9,
                        help='legacy full-controller growth/frontier threshold; '
                             'ignored by the performance controller')
    # Deprecated: LiquidFL now always uses the exact AdaptiveFL aggregation.
    parser.add_argument('--liquid_aggregation_alpha', type=float, default=0.0)
    parser.add_argument('--liquid_kd_decay_start', type=int, default=-1,
                        help='KD cosine-decay start round; -1 with an end of '
                             '-1 selects 30%% of total rounds')
    parser.add_argument('--liquid_kd_decay_end', type=int, default=-1,
                        help='KD cosine-decay end round; -1 with a start of '
                             '-1 selects 80%% of total rounds')
    parser.add_argument('--liquid_kd_final_ratio', type=float, default=0.10,
                        help='final KD gamma divided by the configured gamma')
    # ---- accounting / misc ----
    parser.add_argument('--liquid_macs_coeff', type=float, default=3.0,
                        help='fwd+bwd multiplier for estimated training MACs')
    parser.add_argument(
        '--liquid_eval_interval', type=int, default=10,
        help='interval for detailed controller diagnostics; all five shared '
             'profiles are evaluated on the global test set every round')
    parser.add_argument(
        '--liquid_eval_all_profiles', type=int, default=1,
        help='1 evaluates F0-F4 every round; 0 evaluates only the full F4 '
             'model (useful for large setting sweeps)')
    parser.add_argument('--liquid_run_all', type=int, default=0,
                        help='1: run both backbones and both CIFAR datasets; '
                             '0: respect --model and --dataset (default: '
                             'ResNet34 + CIFAR-10)')
    parser.add_argument(
        '--controller_ablation_variants', type=str,
        default='random,gru,reset',
        help='comma-separated controller variants for '
             'main_controller_ablation.py')
    parser.add_argument(
        '--controller_ablation_datasets', type=str,
        default='cifar10,cifar100',
        help='comma-separated datasets for main_controller_ablation.py')
    parser.add_argument('--liquid_trace_enabled', type=int, default=0,
                        help='1 records passive per-client LiquidFL traces for '
                             'the motivation analysis; 0 has no trace I/O')
    parser.add_argument('--liquid_trace_dir', type=str, default='',
                        help='trace artifact directory; empty selects a path '
                             'below output_dir')
    parser.add_argument('--liquid_trace_checkpoint_interval', type=int,
                        default=0,
                        help='overwrite a restart-safe analysis checkpoint '
                             'every N rounds; 0 saves only the final round')
    parser.add_argument('--liquid_motivation_val_ratio', type=float,
                        default=0.20,
                        help='client-local holdout ratio used only by the '
                             'dedicated motivation experiment entry point')
    # Deprecated (parsed for backward config compatibility, unused by scheme A).
    parser.add_argument('--liquid_adjust_ratio', type=float, default=0.3)
    parser.add_argument('--liquid_probe_ema', type=float, default=0.9)
    parser.add_argument('--liquid_full_profile_boost', type=float, default=4.0)


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
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--generate_data', type=int, default=1, help="whether generate new dataset")
    parser.add_argument('--iid', type=int, default=0, help='whether i.i.d or not')
    parser.add_argument('--noniid_case', type=int, default=4, help="non i.i.d case (1, 2, 3, 4)")
    parser.add_argument('--data_beta', type=float, default=0.6,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--num_classes', type=int, default=100, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
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
    parser.add_argument("--client_chosen_mode", default='available', type=str,
                        help="available / fit / random / RL 客户端选择模式")
    parser.add_argument("--depth_saved", default=[2,3,4], type=int, nargs='*', help="the index of network start channel slim.")  # vgg 采用4-6-8 resnet采用2-3-4 其中resnet是用block为最小单位
    parser.add_argument("--width_ration", default=[0.4, 0.66, 1.0], type=float, nargs='*', help="the info of model ration and model type.")
    parser.add_argument("--client_hetero_ration", default='4:3:3', type=str, help="the info of model ration and model type.")


    # FlexFL
    parser.add_argument("--pretrain", default='0', type=int,
                        help="pretrain rounds to get APOZ")
    parser.add_argument("--T", default=3, type=int,
                        help="nested-subnet distillation temperature")
    parser.add_argument("--gamma", default=10.0, type=float,
                        help="nested-subnet KD weight; 0 gives the CE ablation")
    parser.add_argument("--only", default='1', type=int,
                        help="only use test data or use all data")
    parser.add_argument("--log", default='0', type=int,
                        help="use wandb log or not")
    parser.add_argument("--e", default='0', type=int,
                        help="each pretrain round test APoZ or only test it in final round")
    parser.add_argument("--onlypretrain", default='0', type=int,
                        help="just pretrain to get APoZ but not fed train")
    parser.add_argument("--apoz", default='0', type=int,
                        help="which apoz to use")
    parser.add_argument("--name", default='', type=str,
                        help="wandb display name")
    parser.add_argument("--r", default='0', type=int,
                        help="resource limits (contains uncertainty)")
    parser.add_argument("--decrease", default='0.1', type=float,
                        help="adaptive decrease model")



    args = parser.parse_args()
    return args
'''
# vgg 采用4-6-8 resnet采用2-3-4 其中resnet是用block为最小单位
FedScale:  depth[4, 6, 8] width[0.4, 0.66, 1.0]
HeteroFL: depth[0] width[0.5, 0.71, 1.0]
Decoupled:depth[8] width[0.4, 0.66, 1.0]
'''
