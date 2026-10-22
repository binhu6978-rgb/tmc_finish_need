"""Past-only robust references. Duplicate physical identities receive one vote."""
import copy
from collections import defaultdict
import torch
from torch.nn import functional as F


class SemanticMemory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.anchors = {}

    def snapshot(self):
        return copy.deepcopy(self.anchors)

    def commit(self, round_number, records):
        if round_number == 1:
            return dict(updated=0, reason='independent_initialization_excluded')
        groups = defaultdict(lambda: defaultdict(list))
        for cid, channel, prototypes in records:
            for y, proto in prototypes.items():
                groups[channel, y][cid].append(proto)
        updated = 0; changes = []
        for (channel, y), clients in groups.items():
            if len(clients) < self.cfg['minimum_anchor_clients']:
                continue
            keys = ['probability'] + (['embedding'] if 'embedding' in next(iter(clients.values()))[0] else [])
            current = {}
            for key in keys:
                per_client = [torch.stack([v[key] for v in observations]).mean(0) for observations in clients.values()]
                value = torch.stack(per_client).median(0).values
                current[key] = F.normalize(value, dim=0) if key == 'embedding' else value.clamp_min(0) / value.clamp_min(0).sum().clamp_min(1e-12)
            old = self.anchors.get(channel, {}).get(y)
            if old and round_number - old['round'] <= self.cfg['anchor_max_age']:
                ema = self.cfg['anchor_ema']
                for key in keys:
                    value = ema * old[key] + (1 - ema) * current[key]
                    current[key] = F.normalize(value, dim=0) if key == 'embedding' else value / value.sum()
                changes.append(float(torch.linalg.vector_norm(current['probability'] - old['probability'])))
            current.update(round=round_number, support=len(clients))
            self.anchors.setdefault(channel, {})[y] = current
            updated += 1
        return dict(updated=updated, mean_probability_change=sum(changes)/len(changes) if changes else None,
                    classes_by_channel={str(k): len(v) for k, v in self.anchors.items()})
