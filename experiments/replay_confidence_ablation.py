"""Predict A3 Gates on a FIXED logged trajectory; never trains or predicts accuracy."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from semconsfl.config import SemanticConfig
from semconsfl.diagnostics import distribution
from semconsfl.reliability_memory import ReliabilityMemory


def replay(folder, output):
    folder, output = Path(folder), Path(output)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    config = SemanticConfig(**manifest["semantic_config"])
    memories = {mode: ReliabilityMemory(config) for mode in ("A2", "A3")}
    rows = []
    for expected, line in enumerate((folder / "rounds.jsonl").read_text(encoding="utf-8").splitlines()):
        record = json.loads(line)
        t = record["round"]
        assert t == expected and record["mode"] == "A2"
        old = {mode: memory.snapshot() for mode, memory in memories.items()}
        observations = {mode: [] for mode in memories}
        for slot in record["slots"]:
            classes = [c for c in slot["evidence"]["classes"] if c["reason"] == "valid"]
            support = sum(c["count"] for c in classes)
            a3 = sum(c["count"] * max(0., c["signed_drift"]) for c in classes) / support if support else None
            for mode, anomaly in (("A2", slot["evidence"]["anomaly"]), ("A3", a3)):
                decision = memories[mode].decide(slot["client"], slot["profile"], anomaly, t, old[mode])
                if mode == "A2":
                    assert decision["reason"] == slot["decision_reason"]
                    assert all(decision[k] == slot[k] for k in ("z", "r", "risk", "gamma"))
                rows.append(dict(mode=mode, round=t + 1, slot=slot["slot"], client=slot["client"],
                                 profile=slot["profile"], anomaly=anomaly, **decision))
                observations[mode].append((slot["client"], slot["profile"], anomaly, decision))
        for mode, memory in memories.items():
            memory.update(t, observations[mode])
    summary = {"scope": "Offline prediction on fixed A2 uploads/signatures/anchors; NOT an A3 closed-loop training result or accuracy estimate",
               "source": str(folder.resolve()), "source_git_head": manifest["git_head"], "A2_replay_exact": True,
               "rounds": expected + 1, "modes": {}}
    for mode in memories:
        selected = [r for r in rows if r["mode"] == mode]
        active = [r for r in selected if r["gamma"] < 1]
        summary["modes"][mode] = {"slots": len(selected), "positive_anomaly": sum(r["anomaly"] is not None and r["anomaly"] > 0 for r in selected),
            "valid_anomaly": sum(r["anomaly"] is not None for r in selected), "nonneutral": len(active),
            "first_active_round": min((r["round"] for r in active), default=None),
            "reasons": dict(Counter(r["reason"] for r in selected)),
            "distributions": {k: distribution(r[k] for r in selected) for k in ("anomaly", "z", "r", "risk", "gamma")},
            "profiles": {str(p): {"slots": sum(r["profile"] == p for r in selected),
                "nonneutral": sum(r["profile"] == p for r in active),
                "first_active_round": min((r["round"] for r in active if r["profile"] == p), default=None),
                "gamma": distribution(r["gamma"] for r in selected if r["profile"] == p)} for p in range(7)}}
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with (output / "slots.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({m: {k: v for k, v in s.items() if k not in ("distributions", "profiles")} for m, s in summary["modes"].items()}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    replay(arguments.input, arguments.output)
