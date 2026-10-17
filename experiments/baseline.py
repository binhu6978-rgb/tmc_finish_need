"""Read-only access to the ORIGINAL CIFAR100 partition and experiment settings.

No repartitioning, validation split, augmentation change, download, or cache write.
The newest user instruction supersedes the proposed new validation protocol.
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from torchvision import datasets, transforms

from models.Fed import get_model_list, select_clients
from models.resnet import ResNet18_cifar
from semconsfl.aggregation import coverage
from semconsfl.config import SemanticConfig
from semconsfl.measurement import isolated_rng
from utils.options import args_parser
from utils.set_seed import set_random_seed


BASELINE_COMMIT = "d6354d073ff4d5614a4e6b25fb542a9b43e0dfa8"
BASELINE_FILES = ["main_fed_ori.py", "Algorithm/Training_AdaptiveFL.py", "models/Fed.py",
                  "models/Update.py", "models/resnet.py", "models/test.py", "utils/options.py",
                  "utils/get_datatset_new.py", "utils/HeteroClients.py", "utils/hetero_client_config.py"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["A0", "A1", "A2"], default="A2")
    parser.add_argument("--dataset-root", default="/home/hh/ICML_finish/data")
    parser.add_argument("--partition-path")
    parser.add_argument("--output-dir", required=False)
    parser.add_argument("--semantic-config")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    # Loading the existing split is mandatory. This changes no dataset indices.
    parser.add_argument("--generate_data", type=int, choices=[0], default=0)
    extra, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + remaining
        args = args_parser()  # original defaults, SGD definition, loader batch sizes
    finally:
        sys.argv = old_argv
    args.model, args.dataset, args.algorithm = "resnet", "cifar100", "AdaptiveFL"
    args.num_classes = 100
    args.depth_saved, args.width_ration = [2, 3, 4], [0.4, 0.66, 1.0]
    args.generate_data = 0
    if args.client_chosen_mode != "available":
        raise ValueError("This paired implementation supports the original default available scheduling only")
    if args.num_users != 100 or args.client_hetero_ration != "4:3:3" or args.num_channels != 3:
        raise ValueError("This implementation fixes the requested original 100-client 4:3:3 CIFAR100 setting")
    if not (0 < args.frac <= 1) or args.epochs < 1 or args.local_ep < 1 or args.local_bs < 1 or args.bs < 1:
        raise ValueError("Invalid baseline training configuration")
    if args.gpu >= 0 and not torch.cuda.is_available():
        raise ValueError("Requested CUDA is unavailable; use --gpu -1 explicitly")
    args.device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    if extra.partition_path is None:
        suffix = "iid" if args.iid else f"dirichlet_a{args.data_beta}_bal_minsz10_try500"
        extra.partition_path = f"data/cifar100_U{args.num_users}_seed{args.seed}_{suffix}.json"
    config = SemanticConfig(**json.loads(Path(extra.semantic_config).read_text(encoding="utf-8"))) if extra.semantic_config else SemanticConfig()
    if config.num_classes != 100:
        raise ValueError("CIFAR100 functional signatures require 100 output coordinates")
    return args, extra, config


def load_data(root):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    # Both original train and official test sets, with the original clean transform.
    train = datasets.CIFAR100(root=root, train=True, download=False, transform=transform)
    test = datasets.CIFAR100(root=root, train=False, download=False, transform=transform)
    return train, test


def load_partition(path, args, labels):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if any(payload.get(k) != v for k, v in {"dataset": "cifar100", "num_users": args.num_users,
             "seed": args.seed, "iid": bool(args.iid), "data_beta": args.data_beta}.items()):
        raise ValueError("Partition metadata does not match original experiment arguments")
    users = {int(k): v for k, v in payload["train_data"].items()}
    if set(users) != set(range(args.num_users)):
        raise ValueError("Missing/extra physical clients")
    flat = [i for indices in users.values() for i in indices]
    if (any(type(i) is not int for i in flat) or len(flat) != len(labels)
            or sorted(flat) != list(range(len(labels)))):
        raise ValueError("Partition must cover original train set exactly once, without duplicates/out-of-range indices")
    if min(map(len, users.values())) < args.local_bs:
        raise ValueError("A client has no full original drop_last SGD batch")
    counts = [len(users[i]) for i in range(args.num_users)]
    histograms = [np.bincount(np.asarray(labels)[users[i]], minlength=args.num_classes).tolist()
                  for i in range(args.num_users)]
    return users, {"file": Path(path).name, "sha256": sha256(path),
                   "metadata": {k: v for k, v in payload.items() if k != "train_data"},
                   "total_indices": len(flat), "unique_indices": len(set(flat)),
                   "client_counts": counts, "client_class_histograms": histograms,
                   "min_count": min(counts), "max_count": max(counts),
                   "quantity_balanced": len(set(counts)) == 1,
                   "labels_sha256": content_hash(list(map(int, labels)))}


def initialize_models(args):
    # main_fed_ori constructs this otherwise unused Full model BEFORE get_model_list.
    # Preserve precisely that CPU initialization RNG consumption.
    unused = ResNet18_cifar(num_channels=args.num_channels, num_classes=args.num_classes)
    del unused
    with redirect_stdout(io.StringIO()):
        models, profiles = get_model_list(args)
    if len(models) != 7:
        raise ValueError("Expected seven independent initializations")
    return models, profiles


def dispatch(args):
    profiles = np.random.choice(range(7), max(int(args.frac * args.num_users), 1))
    clients = select_clients(args, profiles, 7)
    return [(int(client), int(profile)) for client, profile in zip(clients, profiles)]


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in state.items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def audit(args, config, partition_report):
    with isolated_rng():
        set_random_seed(args.seed)
        models, profiles = initialize_models(args)
        _, shapes = coverage(models)
        initial = [state_hash(model.state_dict()) for model in models]
        schedules = [dispatch(args) for _ in range(args.epochs)]
    serial_args = {k: str(v) if isinstance(v, torch.device) else v for k, v in vars(args).items()}
    return {"reference_commit": BASELINE_COMMIT,
            "source_sha256": {p: sha256(p) for p in BASELINE_FILES},
            "baseline_args": serial_args, "semantic_config": asdict(config),
            "partition": partition_report, "profiles": profiles, "coverage": shapes,
            "initial_state_sha256": initial, "schedule": schedules,
            "schedule_sha256": content_hash(schedules),
            "evaluation": "Original official test every round, all seven profiles; no new validation split. Test peak is historically exposed, not independent validation.",
            "transform": "ToTensor; Normalize((.5071,.4867,.4408),(.2675,.2565,.2761)); no augmentation",
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                            "device": str(args.device)},
            "training_executed_for_audit": False}
