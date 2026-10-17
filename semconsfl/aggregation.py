"""Original coordinate coverage denominator, gated DELTA numerator only."""
import math

import torch

from models.Fed import Aggregation_AdaptiveFL


def prefix(full, local, key):
    # Exactly the original mapping: weight first two axes, BN/bias first axis.
    if full.dtype != torch.float32 or local.dtype != torch.float32 or full.ndim < 1:
        raise ValueError(f"Unsupported baseline state item: {key}; current ResNet uses float32 vectors/tensors")
    axes = 2 if key.split(".")[-1] == "weight" and full.ndim > 1 else 1
    if (local.ndim != full.ndim or local.device != full.device
            or any(d <= 0 or d > f for d, f in zip(local.shape, full.shape))
            or local.shape[axes:] != full.shape[axes:]):
        raise ValueError(f"Invalid prefix shape/device: {key}")
    return tuple(slice(0, d) for d in local.shape[:axes])


def validate(full, locals_, counts, gammas):
    if not locals_ or not (len(locals_) == len(counts) == len(gammas)):
        raise ValueError("Need aligned, nonempty local uploads/counts/gates")
    if any(not math.isfinite(n) or n <= 0 for n in counts):
        raise ValueError("Invalid aggregation sample count")
    if any(not math.isfinite(g) or not 0 <= g <= 1 for g in gammas):
        raise ValueError("Invalid gate")
    for local in locals_:
        if full.keys() != local.keys():
            raise ValueError("Local/global state keys differ")
        for key, old in full.items():
            prefix(old, local[key], key)
            if not torch.isfinite(old).all() or not torch.isfinite(local[key]).all():
                raise ValueError(f"Nonfinite upload/global state: {key}")


@torch.no_grad()
def aggregate(full, locals_, counts, gammas, round_id):
    validate(full, locals_, counts, gammas)
    if round_id == 0 or all(g == 1.0 for g in gammas):
        # Bitwise golden path, including independent round0 model initialization.
        return Aggregation_AdaptiveFL(locals_, counts, full)
    result = {}
    for key, old in full.items():
        numerator, denominator = torch.zeros_like(old), torch.zeros_like(old)
        for local, n, gamma in zip(locals_, counts, gammas):
            index = prefix(old, local[key], key)
            numerator[index] += (local[key] - old[index]) * (n * gamma)
            denominator[index] += n  # never multiply by gamma
        covered = denominator > 0
        new = old.clone()
        new[covered] += numerator[covered] / denominator[covered]
        result[key] = new
    return result


def coverage(models):
    full = models[-1].state_dict()
    counts, statistics = {}, {"all_profiles": 0, "partial_profiles": 0, "full_only": 0}
    for key, old in full.items():
        count = torch.zeros_like(old, dtype=torch.uint8)
        for model in models:
            count[prefix(old, model.state_dict()[key], key)] += 1
        counts[key] = count
        statistics["all_profiles"] += int((count == len(models)).sum())
        statistics["full_only"] += int((count == 1).sum())
        statistics["partial_profiles"] += int(((count > 1) & (count < len(models))).sum())
    statistics["total_parameters"] = sum(statistics.values())
    return counts, statistics


@torch.no_grad()
def update_diagnostics(old, new, ungated, masks, num_profiles):
    result = {}
    for name, select in (("all_profiles", lambda n: n == num_profiles),
                         ("partial_profiles", lambda n: (n > 1) & (n < num_profiles)),
                         ("full_only", lambda n: n == 1)):
        gated_sq, original_sq = 0.0, 0.0
        for key, theta in old.items():
            mask = select(masks[key])
            gated_sq += float((new[key][mask] - theta[mask]).double().square().sum())
            original_sq += float((ungated[key][mask] - theta[mask]).double().square().sum())
        result[name] = {"update_l2": math.sqrt(gated_sq), "ungated_update_l2": math.sqrt(original_sq),
                        "norm_ratio": math.sqrt(gated_sq / original_sq) if original_sq > 0 else None}
    return result
