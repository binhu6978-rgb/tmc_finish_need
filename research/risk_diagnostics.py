"""Read-only FD-SemConsFL diagnosis. No models are trained or evaluated."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from semconsfl.config import SemanticConfig
from semconsfl.diagnostics import distribution, midranks
from semconsfl.measurement import Signature
from semconsfl.reliability_memory import ReliabilityMemory
from semconsfl.semantic_memory import SemanticMemory

REFERENCE = "b546112e482c4dbdeb6ed49744cf9a8ae64452fc"
SHORT_ROUNDS = 20  # One prespecified horizon; no window/threshold search.
METHODS = ("formal", "ecdf_rank", "short20")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite(v) for v in value]
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(finite(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(finite(rows))


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def rho(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    a, b = np.asarray(pairs).T
    a, b = midranks(a), midranks(b)
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sum(a*a) * np.sum(b*b))**.5
    return float(np.clip(np.sum(a*b) / d, -1, 1)) if d else None


def fixed_effect_residuals(y, *groups):
    """OLS residual projection for additive group effects; no native LAPACK call.

    Local np.linalg.lstsq exits the process even for a 3x2 toy matrix. Alternating
    orthogonal group demeaning is equivalent here and checks both normal equations.
    """
    residual = np.asarray(y, dtype=float).copy()
    groups = [np.asarray(g, dtype=int) for g in groups]
    for iteration in range(10000):
        before = residual.copy()
        for g in groups:
            counts = np.bincount(g)
            sums = np.bincount(g, weights=residual)
            residual -= sums[g] / counts[g]
        if float(np.max(np.abs(before-residual))) < 1e-13:
            break
    else:
        raise RuntimeError("Fixed-effect projection did not converge")
    error = max(float(np.max(np.abs(np.bincount(g,weights=residual)[np.unique(g)] /
                                   np.bincount(g)[np.unique(g)]))) for g in groups)
    assert error < 1e-10
    return residual, iteration+1, error


class ShadowMemory(ReliabilityMemory):
    """Each candidate owns its past calibrated client history, never formal r."""
    def __init__(self, config, method):
        super().__init__(config)
        if method not in METHODS:
            raise ValueError(method)
        self.method = method

    def decide(self, client, profile, anomaly, round_id, snapshot):
        profiles, clients = snapshot
        if self.method == "short20":
            profiles = {p: [r for r in self.recent(records, round_id) if round_id - r[0] <= SHORT_ROUNDS]
                        for p, records in profiles.items()}
        result = super().decide(client, profile, anomaly, round_id, (profiles, clients))
        if self.method != "ecdf_rank" or result["z"] is None:
            return result
        values = np.asarray([r[2] for r in self.recent(profiles.get(profile, []), round_id)])
        percentile = float(((values < anomaly).sum() + .5 * (values == anomaly).sum()) / len(values))
        risk = percentile * result["r"]
        return dict(result, z=percentile, risk=risk, gamma=1 - self.config.gate_strength * risk)


def decode(payload):
    matrix = torch.zeros(100, 100, dtype=torch.float64)
    for c, row in payload["probabilities"].items():
        matrix[int(c)] = torch.tensor(row, dtype=torch.float64) if row is not None else float("nan")
    return Signature(matrix, torch.tensor(payload["counts"], dtype=torch.long), payload["metadata"])


def label_features(hist):
    counts = np.asarray(hist)
    p = counts[counts > 0] / counts.sum()
    return {"n": int(counts.sum()), "labels_present": len(p),
            "label_entropy": float(-(p * np.log(p)).sum() / np.log(100)),
            "majority_fraction": float(p.max()), "supported_classes": int((counts >= 5).sum()),
            "supported_sample_fraction": float(counts[counts >= 5].sum() / counts.sum())}


def sample_range(n):
    return "n<400" if n < 400 else "400<=n<550" if n < 550 else "n>=550"


def replay(folder, mode):
    manifest = read(folder / "manifest.json")
    assert manifest["git_head"] == REFERENCE and manifest["mode"] == mode
    for key, digest in {**manifest["source_sha256"], **manifest["implementation_sha256"]}.items():
        assert sha(key) == digest, "Formal source changed: " + key
    config = SemanticConfig(**manifest["semantic_config"])
    semantic = SemanticMemory(config)
    memories = {name: ShadowMemory(config, name) for name in METHODS}
    rows, shadows, pools, old_client_ledger = [], [], [], {}
    maximum = 0.
    for t, line in enumerate((folder / "rounds.jsonl").open(encoding="utf-8")):
        record = json.loads(line)
        assert record["round"] == t and record["mode"] == mode
        assert [[s["client"], s["profile"]] for s in record["slots"]] == manifest["schedule"][t]
        old_s = semantic.snapshot()
        old = {name: memory.snapshot() for name, memory in memories.items()}
        observations = {name: [] for name in memories}
        sem_obs = []
        duplicate_counts = Counter(s["client"] for s in record["slots"])
        same_profile_counts = Counter((s["client"], s["profile"]) for s in record["slots"])
        for p in range(7):
            pool = memories["formal"].recent(old["formal"][0].get(p, []), t)
            values = [x[2] for x in pool]
            early = [x[2] for x in pool if x[0] + 1 <= 20]
            later = [x[2] for x in pool if x[0] + 1 > 20]
            recent = [x[2] for x in pool if t - x[0] <= SHORT_ROUNDS]
            pools.append({"mode": mode, "round": t+1, "profile": p, "records": len(pool),
                "median": float(np.median(values)) if values else None,
                "iqr": float(np.quantile(values, .75) - np.quantile(values, .25)) if values else None,
                "oldest_round": min((x[0]+1 for x in pool), default=None),
                "mean_age": mean(t-x[0] for x in pool), "early20_records": len(early),
                "early20_fraction": len(early)/len(pool) if pool else None,
                "median_excluding_early20": float(np.median(later)) if later else None,
                "short20_records": len(recent), "short20_median": float(np.median(recent)) if recent else None,
                "current_round_observations_used": 0})
        for slot in record["slots"]:
            client, profile = slot["client"], slot["profile"]
            pre, post = decode(slot["pre"]), decode(slot["post"])
            assert pre.counts.tolist() == manifest["partition"]["client_class_histograms"][client]
            assert int(pre.counts.sum()) == slot["n"]
            evidence = semantic.score(pre, post, t, old_s, confidence_guard=mode != "A3")
            anomaly = evidence["anomaly"]
            logged = slot["evidence"]
            assert anomaly == logged["anomaly"]
            for a, b in zip(evidence["classes"], logged["classes"]):
                omit = {"pre_confidence", "post_confidence", "pre_error_mass", "post_error_mass"} if a["count"] < 5 else set()
                assert {k:v for k,v in a.items() if k not in omit} == {k:v for k,v in b.items() if k not in omit}
            decisions = {}
            for name, memory in memories.items():
                decision = memory.decide(client, profile, anomaly, t, old[name])
                decisions[name] = decision
                if name == "formal":
                    for key, value in decision.items():
                        target = slot["decision_reason"] if key == "reason" else slot[key]
                        assert value == target, (mode, t, slot["slot"], key)
                shadows.append({"mode": mode, "method": name, "round": t+1, "slot": slot["slot"],
                                "client": client, "profile": profile, "anomaly": anomaly, **decision})
                observations[name].append((client, profile, anomaly, decision))
            formal = decisions["formal"]
            valid = [c for c in logged["classes"] if c["reason"] == "valid"]
            positive = [c for c in valid if c["signed_drift"] > 0]
            supported = [c for c in logged["classes"] if c["pre_validity"] == c["post_validity"] == "valid"]
            history = memories["formal"].recent(old["formal"][0].get(profile, []), t)
            values = np.asarray([x[2] for x in history])
            client_past = memories["formal"].recent(old["formal"][1].get(client, []), t)
            ledger = [x for x in old_client_ledger.get(client, []) if t-x[0] <= config.max_age_rounds]
            assert [(a[0], a[1]) for a in ledger] == client_past
            known = formal["z"] is not None
            relation = ("below" if anomaly < formal["profile_median"] else "above" if anomaly > formal["profile_median"] else "equal") if known else "not_calibrated"
            row = {"mode": mode, "round": t+1, "slot": slot["slot"], "client": client, "profile": profile,
                **label_features(pre.counts.tolist()), "sample_range": sample_range(slot["n"]),
                "client_slots_this_round": duplicate_counts[client],
                "same_client_profile_slots_this_round": same_profile_counts[client, profile],
                "repeated_physical_client_this_round": duplicate_counts[client] > 1,
                "observed_classes": sum(c["count"] > 0 for c in logged["classes"]),
                "valid_ba_classes": len(supported), "comparable_classes": len(valid),
                "comparable_samples": evidence["support"], "positive_drift_classes": len(positive),
                "positive_drift_confidence_improved": sum(c["confidence_change"] > config.confidence_tolerance for c in positive),
                "mean_confidence_change": sum(c["count"] * (c.get("post_confidence",0)-c.get("pre_confidence",0)) for c in logged["classes"])/slot["n"],
                "anomaly": anomaly, "relation_to_median": relation,
                "old_percentile": float(((values < anomaly).sum()+.5*(values == anomaly).sum())/len(values)) if anomaly is not None and len(values) else None,
                "old_client_records_always": len(client_past),
                "old_client_positive_records": sum(v>0 for _,v in client_past),
                "old_client_z_mean": mean(v for _,v in client_past),
                "old_client_positive_rounds": ";".join(str(a[0]+1) for a in ledger if a[1]>0),
                "old_client_positive_other_profile": sum(a[1]>0 and profile not in a[2] for a in ledger),
                "last_calibrated_client_gap": t-client_past[-1][0] if client_past else None,
                **formal}
            rows.append(row)
            sem_obs.append((client, pre))
        semantic.update(t, sem_obs)
        for name, memory in memories.items():
            memory.update(t, observations[name])
        # Mirror original duplicate handling to retain provenance, not extra votes.
        by_pair, by_client = defaultdict(list), defaultdict(list)
        if t:
            for c,p,a,d in observations["formal"]:
                if d["z"] is not None:
                    by_pair[c,p].append(d["z"])
            for (c,p), zs in by_pair.items():
                by_client[c].append((p,float(np.mean(zs))))
            for c, ps in by_client.items():
                old_client_ledger[c] = ([x for x in old_client_ledger.get(c,[]) if t-x[0]<=config.max_age_rounds]
                    + [(t,float(np.mean([v for _,v in ps])),[p for p,_ in ps])])[-config.client_capacity:]
    assert t == 99 and len(rows) == 1000
    checkpoint = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
    saved = checkpoint["memories"]
    assert checkpoint["next_round"] == 100 and saved["last_round"] == 99
    assert memories["formal"].profiles == saved["profiles"] and memories["formal"].clients == saved["clients"]
    assert semantic.anchors.keys() == saved["anchors"].keys()
    assert all(t == saved["anchors"][c][0] and torch.equal(v,saved["anchors"][c][1]) for c,(t,v) in semantic.anchors.items())
    return rows, shadows, pools, manifest, {"slots":len(rows), "anomaly_and_decisions_exact":True,
        "final_semantic_and_reliability_memory_exact":True, "class_telemetry_limitation":"Below-support probability rows are omitted in raw signatures; their confidence/error-mass telemetry cannot be replayed and is excluded from equality. No scoring input is excluded."}


def funnel(rows):
    calibrated = [r for r in rows if r["z"] is not None]
    eligible = [r for r in rows if r["reason"] == "valid"]
    counts = {"slots":len(rows), "physical_clients":len({r["client"] for r in rows}),
        "client_rounds":len({(r["round"],r["client"]) for r in rows}),
        "extra_duplicate_slots":len(rows)-len({(r["round"],r["client"]) for r in rows}),
        "class_pairs_possible":len(rows)*100,
        "observed_class_pairs":sum(r["observed_classes"] for r in rows),
        "valid_ba_class_pairs":sum(r["valid_ba_classes"] for r in rows),
        "anchor_comparable_class_pairs":sum(r["comparable_classes"] for r in rows),
        "positive_js_class_pairs":sum(r["positive_drift_classes"] for r in rows),
        "positive_js_confidence_improved":sum(r["positive_drift_confidence_improved"] for r in rows),
        "valid_anomaly":sum(r["anomaly"] is not None for r in rows),
        "positive_anomaly":sum(r["anomaly"] is not None and r["anomaly"]>0 for r in rows),
        "calibrated":len(calibrated), "all_history_eligible":len(eligible),
        "anomaly_below_median":sum(r["relation_to_median"]=="below" for r in rows),
        "anomaly_equal_median":sum(r["relation_to_median"]=="equal" for r in rows),
        "anomaly_above_median":sum(r["relation_to_median"]=="above" for r in rows),
        "z_positive":sum(r["z"] is not None and r["z"]>0 for r in rows),
        "r_positive":sum(r["r"]>0 for r in rows),
        "z_and_r_positive":sum(r["z"] is not None and r["z"]>0 and r["r"]>0 for r in rows),
        "nonneutral":sum(r["gamma"]<1 for r in rows),
        "iqr_floor_hits":sum(r.get("iqr_floor_hit",False) for r in rows),
        "z_one":sum(r["z"]==1 for r in calibrated),
        "z_positive_cold_client":sum(r["z"] is not None and r["z"]>0 and r["reason"]=="insufficient_client_history" for r in rows),
        "z_positive_mature_client_zero_r":sum(r["z"] is not None and r["z"]>0 and r["reason"]=="valid" and r["r"]==0 for r in rows),
        "anomaly_mean":mean(r["anomaly"] for r in rows), "median_mean_calibrated":mean(r.get("profile_median") for r in calibrated),
        "iqr_mean_calibrated":mean(r.get("profile_iqr") for r in calibrated),
        "z_mean":mean(r["z"] for r in rows), "r_mean":mean(r["r"] for r in rows),
        "gamma_mean":mean(r["gamma"] for r in rows), "gamma_min":min(r["gamma"] for r in rows)}
    unique = defaultdict(list)
    for row in rows:
        unique[row["round"], row["client"]].append(row)
    for name, test in (("positive_anomaly",lambda r:r["anomaly"] is not None and r["anomaly"]>0),
                       ("z_positive",lambda r:r["z"] is not None and r["z"]>0),
                       ("r_positive",lambda r:r["r"]>0), ("nonneutral",lambda r:r["gamma"]<1)):
        counts[name+"_client_rounds_any"] = sum(any(test(r) for r in group) for group in unique.values())
    counts["eligible_joint_table"] = {f"z{z}_r{r}":sum((x["z"]>0)==bool(z) and (x["r"]>0)==bool(r) for x in eligible) for z in (0,1) for r in (0,1)}
    zpos = sum(r["z"]>0 for r in eligible)
    rpos = sum(r["r"]>0 for r in eligible)
    counts["eligible_independent_marginals_expected_joint_count"] = zpos*rpos/len(eligible) if eligible else None
    for name,numerator,denominator in (
        ("class_support_fraction_of_observed","valid_ba_class_pairs","observed_class_pairs"),
        ("anchor_comparable_fraction","anchor_comparable_class_pairs","valid_ba_class_pairs"),
        ("positive_js_fraction_of_comparable","positive_js_class_pairs","anchor_comparable_class_pairs"),
        ("positive_anomaly_fraction_of_valid","positive_anomaly","valid_anomaly"),
        ("above_median_fraction_of_calibrated","anomaly_above_median","calibrated"),
        ("z_positive_fraction_of_calibrated","z_positive","calibrated"),
        ("r_positive_fraction_of_history_eligible","r_positive","all_history_eligible"),
        ("joint_fraction_of_history_eligible","z_and_r_positive","all_history_eligible"),
        ("gate_fraction_of_slots","nonneutral","slots"),
        ("gate_fraction_of_client_rounds","nonneutral_client_rounds_any","client_rounds")):
        counts[name]=counts[numerator]/counts[denominator] if counts[denominator] else None
    return counts


def grouped(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    output=[]
    for group, items in sorted(groups.items(),key=lambda kv:str(kv[0])):
        f=funnel(items)
        f.pop("eligible_joint_table")
        output.append({"group":str(group), **f})
    return output


def trends(rows):
    valid = [r for r in rows if r["anomaly"] is not None and r["anomaly"]>0]
    collapsed=defaultdict(list)
    for r in valid: collapsed[r["client"],r["profile"],r["round"]].append(r)
    valid=[dict(group[0],anomaly=mean(r["anomaly"] for r in group),
                mean_confidence_change=mean(r["mean_confidence_change"] for r in group)) for group in collapsed.values()]
    by_pair=defaultdict(list)
    for r in valid:
        by_pair[r["client"],r["profile"]].append(r)
    contributions=defaultdict(lambda:[0.,0.])
    pair_slopes=[]
    for (client,profile), group in by_pair.items():
        # Equalize duplicate slots before temporal statistics.
        by_round=defaultdict(list)
        for r in group: by_round[r["round"]].append(r["anomaly"])
        if len(by_round)<2: continue
        t=np.asarray(list(by_round),dtype=float)
        y=np.log([np.mean(by_round[k]) for k in by_round])
        x=t-t.mean(); y=y-y.mean()
        num,den=float(np.sum(x*y)),float(np.sum(x*x))
        if den:
            contributions[client][0]+=num; contributions[client][1]+=den
            pair_slopes.append({"client":client,"profile":profile,"distinct_rounds":len(by_round),"log_anomaly_slope_per_round":num/den})
    total=np.asarray(list(contributions.values()))
    slope=float(total[:,0].sum()/total[:,1].sum())
    rng=np.random.default_rng(20261017)  # Analysis bootstrap only, no training/seed trials.
    samples=rng.integers(0,len(total),size=(1000,len(total)))
    boot=total[samples].sum(axis=1)
    slopes=boot[:,0]/boot[:,1]
    summary={"within_client_profile_log_slope_per_round":slope,
        "bootstrap_1000_client_cluster_95pct_interval":np.quantile(slopes,[.025,.975]).tolist(),
        "bootstrap_scope":"Resamples physical-client clusters on this single fixed trajectory, not independent FL seeds or causal confidence intervals",
        "repeated_client_profile_pairs":len(pair_slopes),"physical_clients_in_slope":len(total),
        "negative_pair_slopes":sum(r["log_anomaly_slope_per_round"]<0 for r in pair_slopes),
        "round_vs_anomaly_client_profile_round_spearman_descriptive":rho([r["round"] for r in valid],[r["anomaly"] for r in valid])}
    # Exact round and profile fixed effects; collapse residuals to physical clients.
    y=np.log([r["anomaly"] for r in valid])
    residual, iterations, error = fixed_effect_residuals(y,[r["round"] for r in valid],[r["profile"] for r in valid])
    summary["fixed_effect_solver"]={"method":"Alternating group projections (OLS residuals)",
        "iterations":iterations,"max_abs_group_mean_residual":error,
        "environment_issue":"Local NumPy lstsq exits code1 without Python traceback on a 3x2 toy input. No training/environment settings changed."}
    clients=defaultdict(list)
    for r,e in zip(valid,residual): clients[r["client"]].append((r,float(e)))
    client_rows=[]
    for c, entries in clients.items():
        representative=entries[0][0]
        # Client-round equal means stop same-round repeated slots adding extra votes.
        rounds=defaultdict(list)
        for row,e in entries: rounds[row["round"]].append((row["anomaly"],e))
        client_rows.append({"client":c, **{k:representative[k] for k in ("n","labels_present","label_entropy","majority_fraction","supported_classes","supported_sample_fraction")},
            "positive_anomaly_client_profile_round_units":len(entries),"distinct_rounds":len(rounds),
            "mean_anomaly_equal_client_round":mean(np.mean([a for a,e in v]) for v in rounds.values()),
            "mean_log_anomaly_round_profile_adjusted":mean(np.mean([e for a,e in v]) for v in rounds.values())})
    associations=[]
    for feature in ("n","labels_present","label_entropy","majority_fraction","supported_classes","supported_sample_fraction"):
        associations.append({"feature":feature,"physical_clients":len(client_rows),
            "client_level_rho_raw":rho([r[feature] for r in client_rows],[r["mean_anomaly_equal_client_round"] for r in client_rows]),
            "client_level_rho_round_profile_adjusted":rho([r[feature] for r in client_rows],[r["mean_log_anomaly_round_profile_adjusted"] for r in client_rows]),
            "interpretation":"Exploratory association; no iid-slot p-values or causal interpretation"})
    summary["within_profile_round_rho"]={str(p):rho([r["round"] for r in valid if r["profile"]==p],[r["anomaly"] for r in valid if r["profile"]==p]) for p in range(7)}
    summary["anomaly_vs_confidence_improvement_client_profile_round_rho"] = rho([r["anomaly"] for r in valid],[r["mean_confidence_change"] for r in valid])
    return summary,pair_slopes,client_rows,associations


def shadow_summary(rows, shadows):
    result={}
    base={(r["round"],r["slot"]):r for r in rows}
    for method in METHODS:
        items=[r for r in shadows if r["method"]==method]
        eligible=[r for r in items if r["z"] is not None]
        active=[r for r in items if r["gamma"]<1]
        result[method]={"slots":len(items),"calibrated":len(eligible),
            "history_eligible":sum(r["reason"]=="valid" for r in items),
            "z_positive":sum(r["z"]>0 for r in eligible),"r_positive":sum(r["r"]>0 for r in items),
            "nonneutral":len(active),"physical_clients_gated":len({r["client"] for r in active}),
            "client_rounds_gated":len({(r["client"],r["round"]) for r in active}),
            "first_gate_round":min((r["round"] for r in active),default=None),
            "distributions":{k:distribution(r[k] for r in items) for k in ("z","r","risk","gamma")},
            "new_active_while_formal_z_zero":sum(base[r["round"],r["slot"]]["z"]==0 for r in active),
            "new_active_while_formal_gamma_one":sum(base[r["round"],r["slot"]]["gamma"]==1 for r in active),
            "profiles":{str(p):{"slots":sum(r["profile"]==p for r in items),
                "nonneutral":sum(r["profile"]==p for r in active),
                "gamma_min":min(r["gamma"] for r in items if r["profile"]==p)} for p in range(7)}}
        if method=="formal": continue
        groups=defaultdict(lambda:defaultdict(list))
        paired=[]
        for r in items:
            a=base[r["round"],r["slot"]]
            if a["z"] is not None and r["z"] is not None:
                paired.append((a,r))
                groups[r["round"],r["profile"]][r["client"]].append((a["risk"],r["risk"]))
        comparison={"common_calibrated_slots":len(paired),
            "pooled_z_spearman_descriptive":rho([a["z"] for a,b in paired],[b["z"] for a,b in paired]),
            "pooled_risk_spearman_descriptive":rho([a["risk"] for a,b in paired],[b["risk"] for a,b in paired]),
            "within_round_profile_distinct_client_pairs":0,"strict_reversals":0,"ties_broken":0,"ties_created":0}
        for clients in groups.values():
            values=[np.mean(v,axis=0) for v in clients.values()]
            for i,x in enumerate(values):
                for y in values[:i]:
                    a,b=np.sign(x-y)
                    comparison["within_round_profile_distinct_client_pairs"]+=1
                    comparison["strict_reversals"]+=int(a*b<0)
                    comparison["ties_broken"]+=int(a==0 and b!=0)
                    comparison["ties_created"]+=int(a!=0 and b==0)
        result[method]["ranking_vs_formal"]=comparison
    return result


def inventory(root, manifest):
    files=[]
    pilot=None
    for mode in ("A0","A1","A2","A3"):
        folder=root/f"runs/pair100_{mode}_seed1"
        cp=torch.load(folder/"latest.pt",map_location="cpu",weights_only=False)
        files.append({"path":str(folder/"latest.pt"),"bytes":(folder/"latest.pt").stat().st_size,
            "keys":list(cp),"next_round":cp["next_round"],"global_tensors":len(cp["global"]),
            "global_float32_bytes":sum(v.numel()*v.element_size() for v in cp["global"].values()),
            "upload_tensors_saved":False,"pre_round_snapshots_saved":False})
        if mode == "A3":
            # Project ONE original next schedule; no models/data/SGD are invoked.
            from experiments.baseline import dispatch
            python_before, numpy_before = random.getstate(), np.random.get_state()
            try:
                random.setstate(cp["rng"]["python"])
                np.random.set_state(cp["rng"]["numpy"])
                schedule=dispatch(SimpleNamespace(**manifest["baseline_args"]))
            finally:
                random.setstate(python_before); np.random.set_state(numpy_before)
            profiles=[round(p[2]*10**6) for p in manifest["profiles"]]
            pilot={"proposal_only_not_executed":True,"frozen_full_after_round":100,
                "diagnostic_score_round_zero_based":100,"original_next_schedule":schedule,
                "distinct_physical_clients":len({c for c,p in schedule}),
                "learning_rate_at_this_state":manifest["baseline_args"]["lr"]*manifest["baseline_args"]["lr_decay"]**100,
                "local_sgd_steps":sum(5*(manifest["partition"]["client_counts"][c]//50) for c,p in schedule),
                "float32_pre_upload_post_bytes":sum(profiles[p]*4 for c,p in schedule)+profiles[-1]*8,
                "purpose":"Generate NEW uploads conditional on frozen saved Full/Memory state, not reconstruct old uploads, not resume or change epochs fingerprint"}
        del cp
    diagnostics=[]
    for path in sorted((root/"runs/four100_control_20261017/diagnosis").rglob("weights.pt")):
        cp=torch.load(path,map_location="cpu",weights_only=True)
        trace=read(path.parent/"trace.json")
        diagnostics.append({"path":str(path),"bytes":path.stat().st_size,"keys":list(cp),
            "uploads":len(cp["uploads"]),"round_1_based":1,"label":trace["label"],"mode":trace["mode"],
            "pre_full_saved":False,"initial_state_hashes_only":True,
            "upload_float32_bytes":sum(v.numel()*v.element_size() for u in cp["uploads"] for v in u.values())})
        del cp
    profiles=[round(p[2]*10**6) for p in manifest["profiles"]]
    estimates=[]
    for t,schedule in enumerate(manifest["schedule"]):
        upload_bytes=sum(profiles[p]*4 for c,p in schedule)
        estimates.append({"round":t+1,"upload_bytes":upload_bytes,
                          "pre_full_plus_uploads_bytes":upload_bytes+profiles[-1]*4,
                          "pre_full_uploads_post_full_bytes":upload_bytes+profiles[-1]*8})
    return {"formal_checkpoints":files,"separate_round1_diagnostics":diagnostics,
        "formal_per_round_uploads_available":False,"reconstruct_formal_100round_counterfactuals":False,
        "reason":"Only final Full and final Memory/RNG survive in formal checkpoints. Signatures, scalar norms, hashes and final Full do not determine discarded local tensors or old Full states.",
        "limited_round1_exception":"Five independent diagnosis fragments preserve round1 uploads and post Full, but not the formal A2/A3 trajectories or later nonneutral rounds; round1 intentionally has no semantic Gate.",
        "storage_estimates_lossless_float32":{
            "all100_pre_full_uploads_bytes":sum(r["pre_full_plus_uploads_bytes"] for r in estimates),
            "all100_pre_upload_post_bytes":sum(r["pre_full_uploads_post_full_bytes"] for r in estimates),
            "mean_pre_upload_post_per_round_bytes":mean(r["pre_full_uploads_post_full_bytes"] for r in estimates),
            "five_rounds_20_40_60_80_100_bytes":sum(r["pre_full_uploads_post_full_bytes"] for r in estimates if r["round"]%20==0)},
        "per_round_storage":estimates,"minimal_frozen_state_pilot":pilot}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a2",type=Path,default=Path("runs/pair100_A2_seed1"))
    parser.add_argument("--a3",type=Path,default=Path("runs/pair100_A3_seed1"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    root=Path.cwd(); output=args.output.resolve()
    for source in (args.a2.resolve(),args.a3.resolve()):
        if output==source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output must be separate from raw experiments")
    output.mkdir(parents=True,exist_ok=False)
    required=[root/"runs/four100_analysis_seed1"/name for name in
        ("four100_report.md","slots.csv","diagnostics_each_round.csv","gate_profiles.csv","accuracy_each_round.csv")]
    assert all(p.is_file() for p in required)
    provenance={"reference_training_commit":REFERENCE,"analysis_script_sha256":sha(__file__),
        "head_at_execution":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "command":[sys.executable,"-m","research.risk_diagnostics",*sys.argv[1:]],
        "python":sys.version,"numpy":np.__version__,"torch":torch.__version__,
        "no_training_or_model_evaluation":True,"short_profile_window_prespecified":SHORT_ROUNDS,
        "sample_ranges_prespecified":["n<400","400<=n<550","n>=550"],
        "input_hashes":{str(p):sha(p) for p in required}}
    all_rows=[]; all_shadows=[]; all_pools=[]; replay_checks={}; manifests={}
    for mode,folder in (("A2",args.a2),("A3",args.a3)):
        for filename in ("manifest.json","rounds.jsonl","latest.pt"):
            provenance["input_hashes"][str(folder/filename)]=sha(folder/filename)
        rows,shadows,pools,manifest,check=replay(folder,mode)
        all_rows+=rows; all_shadows+=shadows; all_pools+=pools
        replay_checks[mode]=check; manifests[mode]=manifest
    assert manifests["A2"]["schedule"]==manifests["A3"]["schedule"]
    old_slots={}
    with required[1].open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["mode"] in ("A2","A3"):
                old_slots[r["mode"],int(r["round"]),int(r["slot"])]=r
    for r in all_rows:
        old=old_slots[r["mode"],r["round"],r["slot"]]
        for k in ("anomaly","z","r","risk","gamma"):
            assert (None if old[k]=="" else float(old[k]))==r[k]
    a3=[r for r in all_rows if r["mode"]=="A3"]
    a3sh=[r for r in all_shadows if r["mode"]=="A3"]
    summary={"scope":"Descriptive fixed A3 trajectory and strictly past shadow scores; no harmful-update labels, no accuracy/utility claims, no training",
        "replay":replay_checks,"A2":funnel([r for r in all_rows if r["mode"]=="A2"]),"A3":funnel(a3),
        "A3_shadow":shadow_summary(a3,a3sh),"exact_equality_policy":"Exact numerical <, =, > versus stored old median; no tolerance changes the gate threshold"}
    temporal,pair_slopes,clients,associations=trends(a3)
    summary["trends"]=temporal; summary["client_associations"]=associations
    summary["A3_old_percentile_when_formal_z_zero"]=distribution(r["old_percentile"] for r in a3 if r["z"]==0)
    summary["A3_positive_z_blockers"]={"cold_client":sum(r["z"] is not None and r["z"]>0 and r["reason"]=="insufficient_client_history" for r in a3),
        "mature_all_old_z_zero":sum(r["z"] is not None and r["z"]>0 and r["reason"]=="valid" and r["r"]==0 for r in a3),
        "active":sum(r["gamma"]<1 for r in a3)}
    summary["A3_old_r_positive_but_current_z_zero"]=sum(r["r"]>0 and r["z"]==0 for r in a3)
    summary["A3_r_positive_using_only_other_profile_positive_history"]=sum(r["r"]>0 and r["old_client_positive_records"]==r["old_client_positive_other_profile"] for r in a3)
    history_lookup={(r["round"],r["profile"]):r for r in all_pools if r["mode"]=="A3"}
    summary["A3_late_history_lag_slot_weighted"]={name:mean(history_lookup[r["round"],r["profile"]][name] for r in a3 if r["round"]>=81)
        for name in ("median","short20_median","median_excluding_early20","early20_fraction","mean_age","records")}
    summary["A3_formal_profile_history_max_records"]=max(p["records"] for p in all_pools if p["mode"]=="A3")
    write_csv(output/"slots_replayed.csv",all_rows)
    write_csv(output/"shadow_slots.csv",all_shadows)
    write_csv(output/"profile_history_each_round.csv",all_pools)
    for name,key in (("round",lambda r:r["round"]),("profile",lambda r:r["profile"]),
                     ("client",lambda r:r["client"]),("sample_range",lambda r:r["sample_range"]),
                     ("round20_block",lambda r:1+(r["round"]-1)//20)):
        write_csv(output/f"A3_flow_by_{name}.csv",grouped(a3,key))
    write_csv(output/"A3_flow_by_profile_round20_block.csv",grouped(a3,lambda r:(r["profile"],1+(r["round"]-1)//20)))
    write_csv(output/"within_client_profile_trends.csv",pair_slopes)
    write_csv(output/"client_features_and_adjusted_scores.csv",clients)
    write_csv(output/"client_associations.csv",associations)
    write_csv(output/"A3_positive_z_events.csv",[r for r in a3 if r["z"] is not None and r["z"]>0])
    dump(output/"artifact_inventory.json",inventory(root,manifests["A3"]))
    dump(output/"provenance.json",provenance);dump(output/"summary.json",summary)
    print(json.dumps(finite({"A3":summary["A3"],"shadows":{m:{k:v for k,v in d.items() if k not in ("distributions","profiles")} for m,d in summary["A3_shadow"].items()},"trends":temporal}),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
