"""Past profile calibration, then past physical-client persistence; no lookahead."""
import copy
from collections import defaultdict
import math

import numpy as np


class ReliabilityMemory:
    def __init__(self, config):
        self.config = config
        self.profiles = {}  # profile -> [(round, physical client, raw anomaly)]
        self.clients = {}   # client -> [(round, mean calibrated z)]

    def snapshot(self):
        return copy.deepcopy((self.profiles, self.clients))

    def recent(self, records, round_id):
        if any(t >= round_id for t, *_ in records):
            raise ValueError("Current/future history leaked into decision")
        return [r for r in records if round_id - r[0] <= self.config.max_age_rounds]

    def decide(self, client, profile, anomaly, round_id, snapshot):
        profiles, clients = snapshot
        result = {"z": None, "r": 0.0, "risk": 0.0, "gamma": 1.0,
                  "profile_records": 0, "client_records": 0}
        if anomaly is None:
            return dict(result, reason="missing_anomaly")
        if not math.isfinite(anomaly) or anomaly < 0:
            raise ValueError("Invalid anomaly")
        pool = self.recent(profiles.get(profile, []), round_id)
        result["profile_records"] = len(pool)
        if len(pool) < self.config.min_profile_records:
            return dict(result, reason="insufficient_profile_history")
        values = [v for _, _, v in pool]
        q25, b, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        h = max(float(q75 - q25), self.config.iqr_floor)
        z = float(np.clip((anomaly - b) / h, 0, 1))
        result.update(z=z, profile_median=float(b), profile_iqr=h)
        past = self.recent(clients.get(client, []), round_id)
        m = len(past)
        result["client_records"] = m
        if m < self.config.min_client_records:
            return dict(result, reason="insufficient_client_history")
        r = m / (m + self.config.shrinkage) * float(np.mean([v for _, v in past]))
        risk = z * r
        return dict(result, r=r, risk=risk,
                    gamma=1 - self.config.gate_strength * risk, reason="valid")

    def update(self, round_id, observations):
        """One raw vote per (round,client,profile), one z per (round,client).

        Missing calibration is NOT stored as a real zero client observation.
        Real, valid zero anomalies/z do enter the corresponding histories.
        """
        if round_id == 0:
            return
        raw, calibrated = defaultdict(list), defaultdict(list)
        for client, profile, anomaly, decision in observations:
            if anomaly is not None:
                raw[(int(client), int(profile))].append(anomaly)
            if decision["z"] is not None:
                calibrated[(int(client), int(profile))].append(decision["z"])
        raw_by_profile, z_by_client = defaultdict(list), defaultdict(list)
        for (client, profile), values in raw.items():
            raw_by_profile[profile].append((round_id, client, float(np.mean(values))))
        for profile, values in raw_by_profile.items():
            records = self.recent(self.profiles.get(profile, []), round_id)
            records.extend(values)
            self.profiles[profile] = records[-self.config.profile_capacity:]
        for (client, _), values in calibrated.items():
            z_by_client[client].append(float(np.mean(values)))
        for client, values in z_by_client.items():
            records = self.recent(self.clients.get(client, []), round_id)
            records.append((round_id, float(np.mean(values))))
            self.clients[client] = records[-self.config.client_capacity:]
