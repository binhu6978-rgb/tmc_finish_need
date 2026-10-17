"""Independent entry point; no training runs on import or --audit-only.

Example (future authorized training):
python -m experiments.run_semconsfl --mode A2 --dataset-root DATA --output-dir runs/A2
"""
import json
import os
from pathlib import Path
import subprocess

import torch

from experiments.baseline import (audit, content_hash, dispatch, initialize_models,
                                  load_data, load_partition, parse_args, sha256, write_json_new)
from models.Fed import split_model
from models.test import test_img
from semconsfl.aggregation import coverage
from semconsfl.measurement import capture_rng, restore_rng
from semconsfl.trainer import Memories, run_round
from utils.set_seed import set_random_seed


def atomic_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_suffix(".pending")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def read_checkpoint(path, fingerprint, log_path, mode):
    # Our own checkpoint includes Python/NumPy RNG state, hence weights_only=False.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["fingerprint"] != fingerprint or state["mode"] != mode:
        raise ValueError("Checkpoint configuration, source, data, or mode differs")
    rows = [json.loads(line) for line in Path(log_path).read_text(encoding="utf-8").splitlines()]
    if ([row["round"] for row in rows] != list(range(state["next_round"]))
            or state["memories"]["last_round"] != state["next_round"] - 1):
        raise ValueError("Only complete round boundaries can resume; log/checkpoint mismatch")
    return state


def source_hashes():
    paths = [Path("experiments") / name for name in ("__init__.py", "baseline.py", "run_semconsfl.py")]
    paths += sorted(Path("semconsfl").glob("*.py"))
    return {p.as_posix(): sha256(p) for p in paths}


def main(argv=None):
    args, extra, config = parse_args(argv)
    if not extra.output_dir:
        raise ValueError("Specify a new --output-dir (or existing directory with --resume)")
    train, test = load_data(extra.dataset_root)
    users, partition = load_partition(extra.partition_path, args, train.targets)
    manifest = audit(args, config, partition)
    manifest["implementation_sha256"] = source_hashes()
    manifest["mode"] = extra.mode
    if extra.mode == "A3":
        manifest["ablation"] = "Only disable true-class confidence guard: candidate=max(0,signed_drift); all other formulas unchanged"
    manifest["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    # HEAD may acquire a documentation-only commit: source bytes, not git ancestry,
    # determine resume compatibility. Dataset paths do not affect paired settings.
    fingerprint = content_hash({k: v for k, v in manifest.items() if k != "git_head"})
    output = Path(extra.output_dir)
    if extra.audit_only:
        if extra.resume:
            raise ValueError("Audit does not resume training")
        output.mkdir(parents=True, exist_ok=False)
        write_json_new(output / "baseline_manifest.json", manifest)
        print(f"Read-only audit complete: {output / 'baseline_manifest.json'}; no training executed")
        return
    if extra.resume:
        state = read_checkpoint(output / "latest.pt", fingerprint, output / "rounds.jsonl", extra.mode)
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json_new(output / "manifest.json", dict(manifest, fingerprint=fingerprint))
        (output / "rounds.jsonl").touch(exist_ok=False)
        state = None
    set_random_seed(args.seed)
    models, _ = initialize_models(args)
    masks, _ = coverage(models)
    memories = Memories(config)
    start = 0
    if state:
        for model in models:
            model.load_state_dict(split_model(state["global"], model.state_dict()))
        memories.load_state_dict(state["memories"])
        start = state["next_round"]
        restore_rng(state["rng"])
    for round_id in range(start, args.epochs):
        slots = dispatch(args)  # same RNG calls as original available scheduler
        if [list(s) for s in slots] != [list(s) for s in manifest["schedule"][round_id]]:
            raise ValueError("Original schedule RNG diverged from audited replay")
        report = run_round(models, train, users, slots, args, config, memories, extra.mode, round_id, masks)
        # Deliberately preserve original official-test evaluation and BN batching.
        # No validation set, held-out train indices, or testing-based Gate decisions.
        report["official_test"] = [dict(zip(("accuracy", "ce"), test_img(model, test, args))) for model in models]
        with (output / "rounds.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        atomic_checkpoint(output / "latest.pt", {"fingerprint": fingerprint, "mode": extra.mode,
                          "next_round": round_id + 1, "global": models[-1].state_dict(),
                          "memories": memories.state_dict(), "rng": capture_rng()})
        print(f"round={round_id} mode={extra.mode} full_test={report['official_test'][-1]['accuracy']:.4f} gate_mean={report['gate_mean']:.6f}")


if __name__ == "__main__":
    main()
