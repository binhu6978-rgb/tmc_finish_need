#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LiquidFL v2 - Client State Tracker

每个客户端维护：
  observation window（长度 = review_interval）
  committed (r_w, slim_idx) 结构参数
  上一轮 transition（state, rw_logit, I_logits, I_idx）供 REINFORCE 使用
"""

import numpy as np
import torch
from collections import deque

STATE_DIM = 12


class ClientStateTracker:
    def __init__(self, client_id, r_max, r_min,
                 data_size, label_entropy,
                 total_rounds, review_interval=20, num_classes=10):
        self.client_id       = client_id
        self.r_max           = r_max
        self.r_min           = r_min
        self.data_size       = data_size
        self.label_entropy   = label_entropy
        self.total_rounds    = total_rounds
        self.review_interval = review_interval
        self.num_classes     = num_classes

        # observation window
        self.window_eval_acc  = deque(maxlen=review_interval)
        self.window_train_acc = deque(maxlen=review_interval)
        self.window_grad_norm = deque(maxlen=review_interval)

        # committed 结构参数
        self.committed_rw       = r_max       # 连续宽度比例
        self.committed_slim_idx = 2           # 离散剪枝起始层
        self.prev_rw            = r_max       # 上一轮 r_w（用于变化幅度限制）
        self.prev_slim_idx      = 2

        # transition buffer（REINFORCE 用）
        self.last_state    = None
        self.last_rw_logit = None
        self.last_I_logits = None
        self.last_I_idx    = None
        self.last_r_w      = r_max

        # 在线归一化
        # norm_mu/var 只对 11 维（不含 label_entropy）做 EMA 归一化
        self.norm_mu   = np.zeros(STATE_DIM - 1, dtype=np.float32)
        self.norm_var  = np.ones(STATE_DIM - 1,  dtype=np.float32)
        self.norm_init = False
        self.ema_alpha = 0.1

        self.current_round = 0

    # ── 更新 window ──────────────────────────────────────────────

    def update(self, round_idx, eval_acc, train_acc, grad_norm):
        self.current_round = round_idx
        self.window_eval_acc.append(float(eval_acc))
        self.window_train_acc.append(float(train_acc))
        self.window_grad_norm.append(float(grad_norm))

    # ── 状态向量 ─────────────────────────────────────────────────

    def get_state(self, round_idx):
        eval_list  = list(self.window_eval_acc)
        train_list = list(self.window_train_acc)
        grad_list  = list(self.window_grad_norm)

        if not eval_list:
            return torch.zeros(1, STATE_DIM)

        acc_start   = eval_list[0]
        acc_end     = eval_list[-1]
        acc_gain    = acc_end - acc_start
        acc_mean    = float(np.mean(eval_list))
        acc_var     = float(np.var(eval_list))

        train_slope = (float(np.polyfit(range(len(train_list)),
                                        train_list, 1)[0])
                       if len(train_list) >= 2 else 0.0)
        eval_slope  = (float(np.polyfit(range(len(eval_list)),
                                        eval_list, 1)[0])
                       if len(eval_list) >= 2 else 0.0)

        min_len  = min(len(eval_list), len(train_list))
        gap_mean = (float(np.mean([t - e for t, e in
                    zip(list(train_list)[-min_len:],
                        list(eval_list)[-min_len:])]))
                    if min_len > 0 else 0.0)

        log_grad   = float(np.log1p(max(np.mean(grad_list), 0.0)
                                    if grad_list else 0.0))
        log_data   = float(np.log1p(self.data_size))
        curr_scale = float(self.committed_rw)
        progress   = round_idx / max(self.total_rounds, 1)

        # label_entropy 是常数，单独归一化到 [-1, 1]（不走 EMA，避免 tanh=0）
        # max_entropy = ln(10) ≈ 2.303（CIFAR-10），归一化到 [0,1] 再映射
        max_ent = float(np.log(max(self.num_classes, 2)))
        norm_ent = float(np.clip(self.label_entropy / max_ent, 0.0, 1.0))
        norm_ent = 2.0 * norm_ent - 1.0   # 映射到 [-1, 1]

        # 其余 11 维走 EMA 归一化
        raw11 = np.array([
            acc_end, acc_gain,
            acc_mean,  acc_var,
            train_slope, eval_slope,
            gap_mean, log_grad,
            log_data, curr_scale, progress,
        ], dtype=np.float32)
        norm11 = self._normalize(raw11)

        # 拼接成 12 维：[label_entropy_norm, ...11维...]
        full = np.concatenate([[norm_ent], norm11]).astype(np.float32)
        return torch.tensor(full, dtype=torch.float32).unsqueeze(0)

    # ── Interval reward ──────────────────────────────────────────

    def get_interval_reward(self,
                             lambda_acc=0.45,
                             lambda_gain=0.25,
                             lambda_cost=0.15,
                             lambda_gap=0.10,
                             lambda_var=0.05):
        eval_list  = list(self.window_eval_acc)
        train_list = list(self.window_train_acc)

        if not eval_list:
            return 0.0, {}

        acc_end   = eval_list[-1]
        acc_start = eval_list[0]
        acc_gain  = acc_end - acc_start
        acc_var   = float(np.var(eval_list))

        min_len  = min(len(eval_list), len(train_list))
        gap_mean = (float(np.mean([t - e for t, e in
                    zip(list(train_list)[-min_len:],
                        list(eval_list)[-min_len:])]))
                    if min_len > 0 else 0.0)

        # cost：当前 r_w 相对于 r_max 的利用率
        cost = self.committed_rw / max(self.r_max, 1e-6)

        reward = (lambda_acc  * acc_end
                  + lambda_gain * acc_gain
                  - lambda_cost * cost
                  - lambda_gap  * max(gap_mean, 0.0)
                  - lambda_var  * acc_var)

        return float(reward), {
            'acc_end':  acc_end,
            'acc_gain': acc_gain,
            'cost':     cost,
            'gap_mean': gap_mean,
            'acc_var':  acc_var,
        }

    # ── Transition buffer ────────────────────────────────────────

    def save_transition(self, state, r_w, slim_idx,
                        rw_logit, I_logits, I_idx):
        """review round 生成新结构时调用，保存完整 transition"""
        self.last_state    = state
        self.last_r_w      = r_w
        self.last_rw_logit = rw_logit
        self.last_I_logits = I_logits
        self.last_I_idx    = I_idx

    def get_transition(self):
        return (self.last_state, self.last_rw_logit,
                self.last_I_logits, self.last_I_idx, self.last_r_w)

    def has_transition(self):
        return self.last_state is not None and self.last_rw_logit is not None

    # ── 提交新结构 ───────────────────────────────────────────────

    def commit_structure(self, r_w, slim_idx):
        self.prev_rw        = self.committed_rw
        self.prev_slim_idx  = self.committed_slim_idx
        self.committed_rw       = r_w
        self.committed_slim_idx = slim_idx

    def clear_window(self):
        self.window_eval_acc.clear()
        self.window_train_acc.clear()
        self.window_grad_norm.clear()

    # ── 归一化 ───────────────────────────────────────────────────

    def _normalize(self, raw):
        """归一化 raw（11维，不含 label_entropy）"""
        eps = 1e-8
        if not self.norm_init:
            self.norm_mu   = raw.copy()
            self.norm_var  = np.ones_like(raw)
            self.norm_init = True
        else:
            self.norm_mu  = ((1 - self.ema_alpha) * self.norm_mu
                             + self.ema_alpha * raw)
            self.norm_var = ((1 - self.ema_alpha) * self.norm_var
                             + self.ema_alpha * (raw - self.norm_mu) ** 2)
        return np.tanh((raw - self.norm_mu) /
                       np.sqrt(np.maximum(self.norm_var, eps))
                       ).astype(np.float32)


class GlobalStateManager:
    def __init__(self, num_clients,
                 r_max_list, r_min_list,
                 data_sizes, label_entropies,
                 total_rounds, review_interval=20, num_classes=10):
        self.trackers = {
            i: ClientStateTracker(
                client_id=i,
                r_max=r_max_list[i],
                r_min=r_min_list[i],
                data_size=data_sizes[i],
                label_entropy=label_entropies[i],
                total_rounds=total_rounds,
                review_interval=review_interval,
                num_classes=num_classes,
            )
            for i in range(num_clients)
        }

    def update(self, cid, round_idx, eval_acc, train_acc, grad_norm):
        self.trackers[cid].update(round_idx, eval_acc, train_acc, grad_norm)

    def get_state(self, cid, round_idx, device='cpu'):
        return self.trackers[cid].get_state(round_idx).to(device)

    def get_interval_reward(self, cid, **kwargs):
        return self.trackers[cid].get_interval_reward(**kwargs)

    def get_committed(self, cid):
        t = self.trackers[cid]
        return t.committed_rw, t.committed_slim_idx

    def get_prev_rw(self, cid):
        return self.trackers[cid].prev_rw

    def commit_structure(self, cid, r_w, slim_idx):
        self.trackers[cid].commit_structure(r_w, slim_idx)

    def save_transition(self, cid, state, r_w, slim_idx,
                        rw_logit, I_logits, I_idx):
        self.trackers[cid].save_transition(
            state, r_w, slim_idx, rw_logit, I_logits, I_idx)

    def get_transition(self, cid):
        return self.trackers[cid].get_transition()

    def has_transition(self, cid):
        return self.trackers[cid].has_transition()

    def clear_window(self, cid):
        self.trackers[cid].clear_window()

    def get_r_max(self, cid): return self.trackers[cid].r_max
    def get_r_min(self, cid): return self.trackers[cid].r_min
