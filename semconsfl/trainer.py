"""One original AdaptiveFL round plus the optional four-stage observer/gate."""
import copy
import time

import torch

from models.Fed import Aggregation_AdaptiveFL, split_model
from models.Update import LocalUpdate_AdaptiveFL
from semconsfl.aggregation import aggregate, update_diagnostics
from semconsfl.diagnostics import summarize_round
from semconsfl.measurement import measure
from semconsfl.reliability_memory import ReliabilityMemory
from semconsfl.semantic_memory import SemanticMemory


class Memories:
    def __init__(self, config):
        self.semantic = SemanticMemory(config)
        self.reliability = ReliabilityMemory(config)
        self.last_round = -1

    def state_dict(self):
        return copy.deepcopy({"anchors": self.semantic.anchors,
                              "profiles": self.reliability.profiles,
                              "clients": self.reliability.clients,
                              "last_round": self.last_round})

    def load_state_dict(self, state):
        state = copy.deepcopy(state)
        self.semantic.anchors = state["anchors"]
        self.reliability.profiles = state["profiles"]
        self.reliability.clients = state["clients"]
        self.last_round = state["last_round"]


def synchronize_device(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def run_round(models, dataset, users, slots, args, config, memories, mode, round_id, masks):
    if mode not in ("A0", "A1", "A2", "A3"):
        raise ValueError("Unknown experiment arm")
    if round_id != memories.last_round + 1:
        raise ValueError("Round boundary/memory mismatch")
    if not slots:
        raise ValueError("No participation slots")
    old_anchor = memories.semantic.snapshot()
    old_history = memories.reliability.snapshot()
    uploads, counts, pre_observations, history_observations, logs = [], [], [], [], []
    timings = {"measurement_seconds": 0.0, "local_train_seconds": 0.0,
               "server_score_seconds": 0.0}
    for slot_id, (client, profile) in enumerate(slots):
        if client not in users or not 0 <= profile < len(models):
            raise ValueError("Invalid dispatch")
        indices = users[client]
        if len(indices) < args.local_bs:
            raise ValueError("No original SGD batch in local slot")
        # Exactly the original constructor/optimizer/data-loader execution path.
        local = LocalUpdate_AdaptiveFL(args=args, dataset=dataset, idxs=indices)
        model = copy.deepcopy(models[profile]).to(args.device)
        synchronize_device(args.device)
        started = time.perf_counter()
        pre = measure(model, dataset, indices, config, args.device) if mode != "A0" else None
        synchronize_device(args.device)
        timings["measurement_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        weights = local.train(round=round_id, net=model)
        synchronize_device(args.device)
        timings["local_train_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        post = measure(model, dataset, indices, config, args.device) if mode != "A0" else None
        synchronize_device(args.device)
        timings["measurement_seconds"] += time.perf_counter() - started
        log = {"slot": slot_id, "client": client, "profile": profile, "n": len(indices),
               "gamma": 1.0, "reason": "baseline", "risk": 0.0}
        if pre is not None:
            score_started = time.perf_counter()
            # A3 is a research ablation only; the default A0/A1/A2 path is intact.
            if mode == "A3":
                evidence = memories.semantic.score(pre, post, round_id, old_anchor, confidence_guard=False)
            else:
                evidence = memories.semantic.score(pre, post, round_id, old_anchor)
            decision = memories.reliability.decide(client, profile, evidence["anomaly"], round_id, old_history)
            timings["server_score_seconds"] += time.perf_counter() - score_started
            log.update(decision)
            log["decision_reason"] = decision["reason"]
            log.update(evidence=evidence, pre=pre.to_dict(config.min_class_samples),
                       post=post.to_dict(config.min_class_samples))
            if round_id == 0:
                log.update(gamma=1.0, reason="independent_initialization_round0")
            elif mode == "A1":
                log.update(gamma=1.0, reason="observer_only", candidate_reason=decision["reason"])
            pre_observations.append((client, pre))
            history_observations.append((client, profile, evidence["anomaly"], decision))
        uploads.append(copy.deepcopy(weights))
        counts.append(len(indices))  # repeats retain every original aggregation slot
        logs.append(log)
    old = models[-1].state_dict()
    synchronize_device(args.device)
    server_started = time.perf_counter()
    gates = [log["gamma"] for log in logs]
    new = aggregate(old, uploads, counts, gates, round_id)
    ungated = new if all(g == 1 for g in gates) else Aggregation_AdaptiveFL(uploads, counts, old)
    diagnostics = update_diagnostics(old, new, ungated, masks, len(models))
    for model in models:
        model.load_state_dict(split_model(new, model.state_dict()))
    # This is the only write phase; all slots above saw exactly the same old state.
    if mode != "A0":
        memories.semantic.update(round_id, pre_observations)
        memories.reliability.update(round_id, history_observations)
    memories.last_round = round_id
    synchronize_device(args.device)
    timings["aggregation_and_memory_seconds"] = time.perf_counter() - server_started
    timings["server_seconds"] = timings["aggregation_and_memory_seconds"] + timings["server_score_seconds"]
    diagnostics_started = time.perf_counter()
    if mode == "A3":
        gate_diagnostics = summarize_round(logs, config, old_anchor, round_id, len(models), confidence_guard=False)
    else:
        gate_diagnostics = summarize_round(logs, config, old_anchor, round_id, len(models))
    timings["gate_diagnostics_seconds"] = time.perf_counter() - diagnostics_started
    return {"round": round_id, "mode": mode, "slots": logs,
            "duplicate_slots": len(slots) - len(set(client for client, _ in slots)),
            "full_slots": sum(profile == len(models) - 1 for _, profile in slots),
            "gate_min": min(gates), "gate_mean": sum(gates) / len(gates),
            "coverage_update_diagnostics": diagnostics, "gating_diagnostics": gate_diagnostics,
            "timings": timings}
