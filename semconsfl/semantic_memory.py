"""Only old, synchronized, pre-training signatures define functional references."""
import copy
from collections import defaultdict
import math

import torch

from semconsfl.measurement import confusion


def js_divergence(p, q):
    for v in (p, q):
        if not torch.isfinite(v).all() or (v < 0).any() or abs(float(v.sum()) - 1) > 1e-5:
            raise ValueError("JS needs valid probability distributions")
    middle = (p + q) / 2
    def kl(v):
        positive = v > 0
        return (v[positive] * (v[positive] / middle[positive]).log()).sum()
    return min(math.log(2), max(0.0, float((kl(p) + kl(q)) / 2)))


class SemanticMemory:
    def __init__(self, config):
        self.config = config
        self.anchors = {}  # class -> (last contributing round, probability vector)

    def snapshot(self):
        return copy.deepcopy(self.anchors)

    def score(self, pre, post, round_id, snapshot):
        if pre.metadata != post.metadata or not torch.equal(pre.counts, post.counts):
            raise ValueError("Pre/post indices, batches and class counts must match")
        if pre.probabilities.shape != (self.config.num_classes, self.config.num_classes) or post.probabilities.shape != pre.probabilities.shape:
            raise ValueError("Invalid signature matrix shape")
        records, weighted, support = [], 0.0, 0
        for c in range(self.config.num_classes):
            b, b_reason = confusion(pre, c, self.config)
            a, a_reason = confusion(post, c, self.config)
            record = {"class": c, "count": int(pre.counts[c]),
                      "pre_validity": b_reason, "post_validity": a_reason}
            for label, signature in (("pre", pre), ("post", post)):
                row = signature.probabilities[c]
                if int(signature.counts[c]) > 0 and torch.isfinite(row).all():
                    record[label + "_confidence"] = float(row[c])
                    record[label + "_error_mass"] = float(torch.cat((row[:c], row[c + 1:])).sum())
            if b is None or a is None:
                record["reason"] = "invalid_class"
            elif c not in snapshot:
                record["reason"] = "no_anchor"
            elif snapshot[c][0] >= round_id:
                raise ValueError("Current/future anchor leaked into decision")
            elif round_id - snapshot[c][0] > self.config.max_age_rounds:
                record["reason"] = "stale_anchor"
            else:
                anchor = snapshot[c][1]
                drift = js_divergence(a[0], anchor) - js_divergence(b[0], anchor)
                candidate = max(0.0, drift) if a[1] <= b[1] + self.config.confidence_tolerance else 0.0
                record.update(reason="valid", signed_drift=drift,
                              confidence_change=a[1] - b[1], candidate=candidate)
                weighted += candidate * record["count"]
                support += record["count"]
            records.append(record)
        return {"anomaly": weighted / support if support else None,
                "support": support, "classes": records,
                "reason": "valid" if support else "no_comparable_classes"}

    def update(self, round_id, observations):
        """Called AFTER aggregation. Round0 heterogeneous initializations are excluded.

        Each (client,class) first averages its valid pre signatures across slots;
        each physical client then supplies one vote to the coordinate median.
        Median is renormalized to the probability simplex before EMA.
        """
        if round_id == 0:
            return
        grouped = defaultdict(list)
        for client, pre in observations:
            for c in range(self.config.num_classes):
                value, _ = confusion(pre, c, self.config)
                if value is not None:
                    grouped[(int(client), c)].append(value[0])
        by_class = defaultdict(list)
        for (_, c), values in grouped.items():
            by_class[c].append(torch.stack(values).mean(0))
        for c, votes in by_class.items():
            if len(votes) < self.config.min_anchor_clients:
                continue
            robust = torch.quantile(torch.stack(votes), 0.5, dim=0)
            if not torch.isfinite(robust).all() or float(robust.sum()) <= 1e-12:
                continue
            robust = robust / robust.sum()
            previous = self.anchors.get(c)
            if previous is not None and round_id - previous[0] <= self.config.max_age_rounds:
                robust = self.config.anchor_decay * previous[1] + (1 - self.config.anchor_decay) * robust
            self.anchors[c] = (round_id, robust / robust.sum())
