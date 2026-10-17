"""Sparse per-client state for budget-matched five-profile LiquidFL.

Every client has only a lightweight server-side monitor.  A persistent
personalized structure (capacity, fixed mask, and liquid memories) is created
lazily after the Need Gate opens.  All resource tiers are eligible whenever a
sampled non-full profile has genuine headroom under the tier budget.  No
client-specific model weights are kept.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


# These are the final three ResNet stage rates of the five nested APOZ/Flex
# profiles whose realized total-parameter ratios match AdaptiveFL's retained
# five-profile budgets. MobileNet has a different parameter distribution and
# therefore requires its own calibrated rates; reusing the ResNet rates would
# silently make all four reduced MobileNet profiles too small.
RESNET_PROFILE_CAPACITIES = (
    (1.00, 0.6061780233, 0.2109216694),  # F0: ~0.2271 params
    (1.00, 1.0000000000, 0.3485992688),  # F1: ~0.4794 params
    (1.00, 1.0000000000, 0.3989558705),  # F2: ~0.5032 params
    (1.00, 1.0000000000, 0.6585097051),  # F3: ~0.6692 params
    (1.00, 1.0000000000, 1.0000000000),  # F4: immutable full model
)

# Flat rates for MobileNet groups 1/2 were calibrated against the same
# realized parameter ratios as the ResNet pool. Separate CIFAR-10/100 values
# compensate for the materially larger classifier share in the small CIFAR-100
# submodels. The maximum absolute ratio error is below 0.23 percentage points.
MOBILENET_PROFILE_CAPACITIES = {
    10: (
        (1.00, 0.4377929688, 0.4377929688),
        (1.00, 0.6743652344, 0.6743652344),
        (1.00, 0.6933837891, 0.6933837891),
        (1.00, 0.8121337891, 0.8121337891),
        (1.00, 1.0000000000, 1.0000000000),
    ),
    100: (
        (1.00, 0.4257324219, 0.4257324219),
        (1.00, 0.6683349609, 0.6683349609),
        (1.00, 0.6873535156, 0.6873535156),
        (1.00, 0.8061035156, 0.8061035156),
        (1.00, 1.0000000000, 1.0000000000),
    ),
}

# Backward-compatible alias for callers that explicitly mean the ResNet pool.
PROFILE_CAPACITIES = RESNET_PROFILE_CAPACITIES
PROFILE_NAMES = ('F0', 'F1', 'F2', 'F3', 'F4')
PROFILE_PROBABILITIES = (2.0 / 7.0, 1.0 / 7.0, 2.0 / 7.0,
                         1.0 / 7.0, 1.0 / 7.0)
PROFILE_TARGET_COSTS = (
    0.22713551, 0.47940375, 0.50322935, 0.66920635, 1.0)


def profile_capacities_for(model_name, num_classes):
    """Return the budget-calibrated nested pool for one backbone/dataset."""
    model_name = str(model_name).lower()
    if model_name == 'resnet':
        return RESNET_PROFILE_CAPACITIES
    if model_name == 'mobilenet':
        num_classes = int(num_classes)
        if num_classes not in MOBILENET_PROFILE_CAPACITIES:
            raise ValueError(
                'MobileNet LiquidFL profiles are calibrated only for '
                '10 or 100 classes; got {}'.format(num_classes))
        return MOBILENET_PROFILE_CAPACITIES[num_classes]
    raise ValueError('unsupported LiquidFL backbone: {!r}'.format(model_name))


def profile_capacity(profile_idx, capacities=PROFILE_CAPACITIES):
    profile_idx = int(profile_idx)
    if profile_idx < 0 or profile_idx >= len(capacities):
        raise ValueError(f'invalid five-profile index: {profile_idx}')
    return torch.tensor(capacities[profile_idx], dtype=torch.float32)


def profile_fixed_mask(profile_idx, capacities=PROFILE_CAPACITIES):
    """Coordinates equal to one in the anchor remain exactly one forever."""
    return profile_capacity(profile_idx, capacities).ge(1.0 - 1e-7)


@dataclass
class LiquidClientState:
    client_id: int
    tier: str
    trajectory: torch.Tensor

    # Lightweight monitoring exists for every client.
    participation_count: int = 0
    last_participation_round: int = -1
    stagnation_ema: float = 0.0
    need_streak: int = 0
    need_streak_by_profile: Dict[int, int] = field(default_factory=dict)
    reuse_remaining: int = 0
    reuse_uses: int = 0
    exploration_uses: int = 0
    last_need: float = 0.0
    last_gate_open: bool = False
    probe_loss_ema: Dict[int, float] = field(default_factory=dict)
    probe_ema_count: Dict[int, int] = field(default_factory=dict)

    # Allocated only for clients promoted by the Need Gate.
    capacity: Optional[torch.Tensor] = None
    anchor_profile: Optional[int] = None
    fixed_mask: Optional[torch.Tensor] = None
    budget: Optional[float] = None
    fast_memory: Optional[torch.Tensor] = None
    slow_memory: Optional[torch.Tensor] = None
    promotion_round: int = -1
    personalized_uses: int = 0

    @property
    def personalized(self):
        return self.capacity is not None


class LiquidStateManager:
    """Client monitors plus lazily allocated personalized structures."""

    def __init__(self, num_clients, hetero_ratio, hidden_dim, projector,
                 profile_capacities=PROFILE_CAPACITIES,
                 performance_mode=False):
        if isinstance(hetero_ratio, str):
            parts = [float(x) for x in hetero_ratio.split(':')]
        else:
            parts = [float(x) for x in hetero_ratio]
        if len(parts) != 3 or any(x <= 0.0 for x in parts):
            raise ValueError('hetero_ratio must contain three positive parts')
        parts = [x / sum(parts) for x in parts]
        weak_end = int(num_clients * parts[0])
        medium_end = int(num_clients * (parts[0] + parts[1]))

        self.hidden_dim = int(hidden_dim)
        self.performance_mode = bool(performance_mode)
        self.projector = projector
        self.profile_capacities = tuple(
            tuple(float(x) for x in capacity)
            for capacity in profile_capacities)
        if len(self.profile_capacities) != 5:
            raise ValueError('LiquidFL requires exactly five profile capacities')
        tier_ceiling = {
            'weak': self.profile_capacities[1],
            'medium': self.profile_capacities[3],
            'strong': self.profile_capacities[4],
        }
        self.tier_budget = {
            tier: float(projector.exact_cost(tier_ceiling[tier]))
            for tier in ('weak', 'medium', 'strong')
        }
        self.states = {}
        for client_id in range(int(num_clients)):
            tier = ('weak' if client_id < weak_end else
                    'medium' if client_id < medium_end else 'strong')
            self.states[client_id] = LiquidClientState(
                client_id=client_id,
                tier=tier,
                trajectory=torch.zeros(4, dtype=torch.float32),
            )

    def get(self, client_id):
        return self.states[int(client_id)]

    def client_ids(self, tier=None, personalized_only=False):
        result = []
        for state in self.states.values():
            if tier is not None and state.tier != tier:
                continue
            if personalized_only and not state.personalized:
                continue
            result.append(state.client_id)
        return sorted(result)

    def tier_of(self, client_id):
        return self.get(client_id).tier

    def has_actionable_headroom(self, state, profile_idx):
        """Whether this anchor can change under the client's tier budget."""
        profile_idx = int(profile_idx)
        # F4 already equals the full model and has no structural headroom.
        # Strong clients remain eligible when they receive F0--F3.
        if profile_idx == 4:
            return False
        anchor = profile_capacity(profile_idx, self.profile_capacities)
        mask = self.fixed_mask_for(profile_idx)
        budget = self.tier_budget[state.tier]
        if self.projector.exact_cost(anchor) > budget + 1e-12:
            return False
        upper = anchor.clone()
        upper[~mask] = 1.0
        feasible = self.projector.project_to_budget_fixed(
            upper, budget, mask, self.projector.min_capacity)
        return (self.projector.hard_signature(feasible)
                != self.projector.hard_signature(anchor))

    def fixed_mask_for(self, profile_idx):
        """Return immutable stages for one activation anchor.

        All five ResNet/MobileNet profiles keep the first controlled stage at
        full width.  The performance controller preserves that common stem
        while allowing stages 2/3 to exchange capacity even when a base
        profile initially places stage 2 at width one.
        """
        if self.performance_mode and int(profile_idx) != 4:
            return torch.tensor([True, False, False], dtype=torch.bool)
        return profile_fixed_mask(profile_idx, self.profile_capacities)

    def activate(self, state, profile_idx, round_idx):
        """Create a personalized structure only after a successful gate."""
        if state.personalized:
            return False
        if not self.has_actionable_headroom(state, profile_idx):
            return False
        anchor = profile_capacity(profile_idx, self.profile_capacities)
        state.capacity = anchor.clone()
        state.anchor_profile = int(profile_idx)
        state.fixed_mask = self.fixed_mask_for(profile_idx)
        state.budget = self.tier_budget[state.tier]
        state.fast_memory = torch.zeros(
            1, self.hidden_dim, dtype=torch.float32)
        state.slow_memory = torch.zeros(
            1, self.hidden_dim, dtype=torch.float32)
        state.promotion_round = int(round_idx)
        return True

    @property
    def personalized_count(self):
        return sum(1 for state in self.states.values() if state.personalized)


class BaselineTracker:
    """EMA loss baseline keyed by a comparable profile/capacity bucket."""

    def __init__(self, ema_decay=0.9, step=0.10):
        self.ema_decay = float(ema_decay)
        self.step = float(step)
        self.mean = {}
        self.sq = {}
        self.count = {}

    def profile_key(self, tier, profile_idx):
        return ('base', str(tier), int(profile_idx))

    def bucket_key(self, tier, capacity):
        values = torch.as_tensor(capacity).detach().cpu().reshape(3).tolist()
        return (
            'personalized', str(tier),
            int(round(values[0] / self.step)),
            int(round(values[1] / self.step)),
            int(round(values[2] / self.step)),
        )

    def stats(self, key):
        if self.count.get(key, 0) == 0:
            return 0.0, 0.0, 0
        mean = self.mean[key]
        variance = max(self.sq[key] - mean * mean, 0.0)
        return mean, variance ** 0.5, self.count[key]

    def update(self, key, probe_loss):
        value = float(probe_loss)
        if key not in self.count:
            self.mean[key] = value
            self.sq[key] = value * value
            self.count[key] = 1
            return
        beta = self.ema_decay
        self.mean[key] = beta * self.mean[key] + (1.0 - beta) * value
        self.sq[key] = beta * self.sq[key] + (1.0 - beta) * value * value
        self.count[key] += 1
