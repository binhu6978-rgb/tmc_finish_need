"""Read-only summaries of already computed evidence; never feed decisions.

Class counts use (slot,class) pairs; score/rank distributions use actual slots.
Physical-client counts are deduplicated separately. Missing denominators are null.
"""
from collections import Counter, defaultdict
import math

import numpy as np


def fraction(count, total):
    return count / total if total else None


def distribution(values):
    """Exact zero/sign counts; finite observations only, missing stays missing."""
    values = list(values)
    present = np.asarray([float(v) for v in values if v is not None], dtype=float)
    if not np.isfinite(present).all():
        raise ValueError("Nonfinite diagnostic observation")
    count = len(present)
    result = {"total": len(values), "valid": count, "missing": len(values) - count,
              "coverage": fraction(count, len(values))}
    for name, predicate in (("zero", present == 0), ("positive", present > 0), ("negative", present < 0)):
        number = int(predicate.sum())
        result[name + "_count"] = number
        result[name + "_fraction"] = fraction(number, count)
    result.update(mean=float(present.mean()) if count else None,
                  min=float(present.min()) if count else None,
                  max=float(present.max()) if count else None,
                  quantiles={str(q): float(np.quantile(present, q)) if count else None
                             for q in (.05, .25, .5, .75, .95)})
    return result


def midranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def rank_difference(formal, candidate):
    paired = [(a, b) for a, b in zip(formal, candidate) if a is not None and b is not None]
    if not paired:
        return {"paired_slots": 0, "spearman": None, "mean_absolute_rank_shift": None,
                "max_absolute_rank_shift": None, "pair_count": 0,
                "strict_order_reversals": 0, "changed_tie_pairs": 0}
    a, b = np.asarray(paired, dtype=float).T
    ra, rb = midranks(a), midranks(b)
    ca, cb = ra - ra.mean(), rb - rb.mean()
    norm = math.sqrt(float(np.dot(ca, ca) * np.dot(cb, cb)))
    reversals, changed_ties = 0, 0
    for i in range(len(a)):
        for j in range(i):
            sa, sb = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            reversals += int(sa * sb < 0)
            changed_ties += int((sa == 0) != (sb == 0))
    return {"paired_slots": len(a), "spearman": float(np.clip(np.dot(ca, cb) / norm, -1, 1)) if norm else None,
            "mean_absolute_rank_shift": float(np.abs(ra - rb).mean()),
            "max_absolute_rank_shift": float(np.abs(ra - rb).max()),
            "pair_count": len(a) * (len(a) - 1) // 2,
            "strict_order_reversals": reversals, "changed_tie_pairs": changed_ties}


def summarize_round(slots, config, old_anchors, round_id, num_profiles, *, confidence_guard=True):
    """Observe logs and the old snapshot without writes, forward passes or RNG calls."""
    observed = [slot for slot in slots if "evidence" in slot]
    records = [record for slot in observed for record in slot["evidence"]["classes"]]
    present = [record for record in records if record["count"] > 0]
    valid_ba = [record for record in records
                if record["pre_validity"] == "valid" and record["post_validity"] == "valid"]
    comparable = [record for record in records if record["reason"] == "valid"]
    positive = [record for record in comparable if record["signed_drift"] > 0]
    improved = [record for record in positive if record["confidence_change"] > 0]
    would_protect = [record for record in positive if record["confidence_change"] > config.confidence_tolerance]
    protected = would_protect if confidence_guard else []
    # Positive drift alone and signed mean are diagnostics on the SAME comparable
    # classes/counts as formal anomaly. They never become a calibration/Gate input.
    formal, unprotected, signed = [], [], []
    by_client = defaultdict(list)
    for slot in observed:
        evidence = slot["evidence"]
        classes = [r for r in evidence["classes"] if r["reason"] == "valid"]
        support = sum(r["count"] for r in classes)
        formal.append(evidence["anomaly"])
        unprotected.append(sum(r["count"] * max(0., r["signed_drift"]) for r in classes) / support if support else None)
        signed.append(sum(r["count"] * r["signed_drift"] for r in classes) / support if support else None)
        if evidence["anomaly"] is not None:
            by_client[slot["client"]].append(evidence["anomaly"])
    all_clients = set(slot["client"] for slot in slots)
    positive_clients = sum(any(a > 0 for a in values) for values in by_client.values())
    zero_clients = sum(all(a == 0 for a in values) for values in by_client.values())
    reasons = Counter(slot.get("decision_reason", slot.get("candidate_reason", slot["reason"])) for slot in observed)
    calibrated = [slot for slot in observed if slot.get("z") is not None]
    floor_hits = sum(bool(slot["iqr_floor_hit"]) for slot in calibrated)
    z = [slot["z"] for slot in calibrated]
    old_anchor_count = sum(0 <= t < round_id and round_id - t <= config.max_age_rounds
                           for t, _ in old_anchors.values())
    total_comparable = len(comparable)
    result = {
        "available": bool(observed), "observed_slots": len(observed),
        "class_pairs": {
            "possible": len(records), "observed": len(present), "valid_ba": len(valid_ba),
            "valid_ba_fraction_of_observed": fraction(len(valid_ba), len(present)),
            "valid_ba_fraction_of_possible": fraction(len(valid_ba), len(records)),
            "comparable_old_anchor": total_comparable,
            "comparable_fraction_of_valid_ba": fraction(total_comparable, len(valid_ba)),
            "positive_drift": len(positive),
            "positive_drift_fraction": fraction(len(positive), total_comparable),
            "negative_drift": sum(r["signed_drift"] < 0 for r in comparable),
            "negative_drift_fraction": fraction(sum(r["signed_drift"] < 0 for r in comparable), total_comparable),
            "positive_drift_confidence_increased": len(improved),
            "confidence_increased_fraction_of_positive": fraction(len(improved), len(positive)),
            "confidence_increased_fraction_of_comparable": fraction(len(improved), total_comparable),
            "confidence_protected": len(protected),
            "protected_fraction_of_positive": fraction(len(protected), len(positive)),
            "protected_fraction_of_comparable": fraction(len(protected), total_comparable),
        },
        "old_anchor_classes": {"valid": old_anchor_count, "total": config.num_classes,
                               "coverage": fraction(old_anchor_count, config.num_classes)},
        # A physical client can have several profile-specific anomalies: classify
        # as positive if ANY valid slot is positive, zero if ALL valid slots zero.
        "physical_clients": {"total": len(all_clients), "with_valid_anomaly": len(by_client),
                             "missing_anomaly": len(all_clients - by_client.keys()),
                             "any_positive_anomaly": positive_clients, "all_valid_anomalies_zero": zero_clients,
                             "positive_fraction_of_valid": fraction(positive_clients, len(by_client)),
                             "zero_fraction_of_valid": fraction(zero_clients, len(by_client))},
        "decision_reasons": dict(reasons),
        "insufficient_profile_slots": reasons["insufficient_profile_history"],
        "insufficient_client_slots": reasons["insufficient_client_history"],
        "iqr_floor": {"calibrated_slots": len(calibrated), "hits": floor_hits,
                      "hit_fraction": fraction(floor_hits, len(calibrated))},
        "z": {"valid": len(z), "missing": len(observed) - len(z),
              "zero_fraction": fraction(sum(v == 0 for v in z), len(z)),
              "interior_fraction": fraction(sum(0 < v < 1 for v in z), len(z)),
              "one_fraction": fraction(sum(v == 1 for v in z), len(z))},
        "r": distribution(slot["r"] for slot in observed),
        "risk": distribution(slot["risk"] for slot in observed),
        "gamma": distribution(slot["gamma"] for slot in slots),
        "nonneutral_gate_slots": sum(slot["gamma"] < 1 for slot in slots),
        "full_profile": num_profiles - 1,
        "profiles": {},
        "anomaly_comparison": {
            "formal": distribution(formal), "positive_without_confidence_guard": distribution(unprotected),
            "signed_mean": distribution(signed),
            "unguarded_vs_formal_ranks": rank_difference(formal, unprotected),
            "signed_vs_formal_ranks": rank_difference(formal, signed)},
    }
    for profile in range(num_profiles):
        rows = [slot for slot in slots if slot["profile"] == profile]
        result["profiles"][str(profile)] = {"slots": len(rows),
                                           "nonneutral_gate_slots": sum(r["gamma"] < 1 for r in rows),
                                           "gamma": distribution(r["gamma"] for r in rows)}
    if not confidence_guard:
        result["confidence_guard_enabled"] = False
        result["class_pairs"]["confidence_would_protect"] = len(would_protect)
    return result
