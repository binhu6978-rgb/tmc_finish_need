"""Support/stability observations, not an assumed harmfulness estimator."""
from collections import defaultdict
import copy
import statistics


class ReliabilityMemory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.clients = defaultdict(list)
        self.profiles = defaultdict(list)

    def snapshot(self):
        return copy.deepcopy((dict(self.clients), dict(self.profiles)))

    def read(self, snapshot, cid, profile, round_number):
        def report(items):
            active = [(r, v) for r, v in items if 0 < round_number-r <= self.cfg['history_max_age']]
            values = [v for _, v in active]
            return dict(support=len(values), ready=len(values) >= self.cfg['history_min_rounds'],
                        age=round_number-active[-1][0] if active else None,
                        mean=statistics.mean(values) if values else None,
                        std=statistics.pstdev(values) if values else None)
        return dict(client=report(snapshot[0].get(cid, [])),
                    client_profile=report(snapshot[1].get((cid, profile), [])))

    def commit(self, round_number, observations):
        if round_number == 1:
            return
        by_client = defaultdict(list); by_profile = defaultdict(list)
        for cid, profile, value in observations:
            if value is not None:
                by_client[cid].append(value); by_profile[cid, profile].append(value)
        for key, values in by_client.items():
            self.clients[key].append((round_number, statistics.mean(values)))
        for key, values in by_profile.items():
            self.profiles[key].append((round_number, statistics.mean(values)))
        for bank in (self.clients, self.profiles):
            for key in bank:
                bank[key] = [(r, v) for r, v in bank[key] if round_number-r <= self.cfg['history_max_age']]
