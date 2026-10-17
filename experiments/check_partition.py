"""In-memory replay of the ORIGINAL unbalanced partition generator. No training."""
import argparse
import json
from pathlib import Path

from experiments.baseline import content_hash, load_data, load_partition, parse_args, sha256
from semconsfl.measurement import isolated_rng
from utils.get_datatset_new import dirichlet_split_unbalanced


def compare_partitions(cached, regenerated):
    if cached.keys() != regenerated.keys():
        raise ValueError("Different physical client keys")
    clients = {}
    for client in sorted(cached):
        a, b = cached[client], regenerated[client]
        clients[str(client)] = {"sequence_equal": a == b, "set_equal": set(a) == set(b),
                                "cached_count": len(a), "regenerated_count": len(b),
                                "symmetric_difference_count": len(set(a) ^ set(b))}
    ordered_a = {str(i): cached[i] for i in sorted(cached)}
    ordered_b = {str(i): regenerated[i] for i in sorted(regenerated)}
    sets_a = {str(i): sorted(cached[i]) for i in sorted(cached)}
    sets_b = {str(i): sorted(regenerated[i]) for i in sorted(regenerated)}
    return {"clients": clients,
            "exact_sequence_equal_clients": sum(v["sequence_equal"] for v in clients.values()),
            "unordered_set_equal_clients": sum(v["set_equal"] for v in clients.values()),
            "all_sequences_equal": ordered_a == ordered_b,
            "all_sets_equal": sets_a == sets_b,
            "cached_ordered_sha256": content_hash(ordered_a),
            "regenerated_ordered_sha256": content_hash(ordered_b),
            "cached_sets_sha256": content_hash(sets_a),
            "regenerated_sets_sha256": content_hash(sets_b)}


def verify_partition(train, args, partition_path):
    before = sha256(partition_path)
    cached, audit = load_partition(partition_path, args, train.targets)
    if args.iid:
        raise ValueError("This review checks original non-IID Dirichlet partition")
    # The original get_dataset forces balanced=False and uses these defaults.
    settings = {"num_clients": args.num_users, "alpha": args.data_beta, "seed": args.seed,
                "min_size": 10, "max_retry": 500, "min_classes": 0}
    with isolated_rng():
        lists = dirichlet_split_unbalanced(train, **settings)
    regenerated = {i: lists[i] for i in range(args.num_users)}
    result = compare_partitions(cached, regenerated)
    arm_hashes = {}
    for mode in ("A0", "A1", "A2"):
        arm_args, extra, _ = parse_args(["--mode", mode, "--gpu", "-1", "--seed", str(args.seed),
                                       "--data_beta", str(args.data_beta), "--partition-path", str(partition_path)])
        arm_users, _ = load_partition(extra.partition_path, arm_args, train.targets)
        arm_hashes[mode] = content_hash({str(i): arm_users[i] for i in sorted(arm_users)})
    after = sha256(partition_path)
    if before != after:
        raise ValueError("Partition file changed during read-only audit")
    return dict(result, generator="utils.get_datatset_new.dirichlet_split_unbalanced",
                generator_source_sha256=sha256("utils/get_datatset_new.py"),
                generator_settings=settings, cached_file=Path(partition_path).name,
                cached_file_sha256=before, cached_file_sha256_after=after,
                metadata=audit["metadata"], min_client_count=audit["min_count"],
                max_client_count=audit["max_count"], labels_sha256=audit["labels_sha256"],
                arm_ordered_sha256=arm_hashes,
                all_arms_read_same_partition=len(set(arm_hashes.values())) == 1,
                training_executed=False, partition_files_written=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--partition-path", default="data/cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json")
    parser.add_argument("--output", required=True)
    parsed = parser.parse_args(argv)
    # This audit is specifically the original alpha=.3/seed=1/100-client setting.
    args, _, _ = parse_args(["--gpu", "-1"])
    train, _ = load_data(parsed.dataset_root)
    report = verify_partition(train, args, parsed.partition_path)
    output = Path(parsed.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ("all_sequences_equal", "all_sets_equal",
                                           "all_arms_read_same_partition", "cached_file_sha256")}, indent=2))


if __name__ == "__main__":
    main()
