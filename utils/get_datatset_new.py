# get_dataset_new.py
# 接口和旧版 get_dataset.py 完全兼容：
#   - 支持 args.noniid_case（透明忽略，统一用 Dirichlet）
#   - 在函数内设置 args.num_classes / args.num_channels
#   - dataset 名称同时支持旧版写法（TinyImagenet）和新版（tiny_imagenet）
#   - _targets_array 兼容自定义数据集（无 .targets 时 fallback）
#   - TinyImageNet 自动添加 .targets 属性
import os
import json
import numpy as np
from torchvision import datasets, transforms
from utils.tinyimagenet import TinyImageNet


# -------------------------
# Record I/O
# -------------------------
def _build_record_path(cfg) -> str:
    os.makedirs("data", exist_ok=True)
    min_size    = int(getattr(cfg, "min_size", 10))
    min_classes = int(getattr(cfg, "min_classes", 0))
    max_retry   = int(getattr(cfg, "max_retry", 500))

    file = os.path.join(
        "data",
        f"{cfg.dataset}_U{cfg.num_users}_seed{cfg.seed}"
    )
    if bool(cfg.iid):
        file += "_iid"
    else:
        file += f"_dirichlet_a{float(cfg.data_beta)}_unbal"
        file += f"_minsz{min_size}"
        if min_classes > 0:
            file += f"_minc{min_classes}"
        file += f"_try{max_retry}"
    return file + ".json"


def _save_record(path, cfg, dict_users, extra_meta=None):
    payload = {
        "dataset":    cfg.dataset,
        "num_users":  int(cfg.num_users),
        "iid":        bool(cfg.iid),
        "data_beta":  float(cfg.data_beta),
        "seed":       int(cfg.seed),
        "extra_meta": extra_meta or {},
        "train_data": dict_users,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_record(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    train_data = {int(k): v for k, v in payload["train_data"].items()}
    return train_data, payload.get("extra_meta", {})


# -------------------------
# Dataset prepare
# -------------------------
def _normalize_dataset_name(name: str) -> str:
    """统一数据集名称，兼容旧版和新版写法"""
    n = name.lower().replace("-", "_").replace(" ", "_")
    if n in ("tinyimagenet", "tiny_imagenet", "tiny imagenet"):
        return "tiny_imagenet"
    return n   # cifar10, cifar100, mnist, ...


def prepare_cifar10(data_dir='./data'):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
    ])
    root = os.path.join(data_dir, 'cifar10')
    return (datasets.CIFAR10(root, train=True, download=True, transform=t),
            datasets.CIFAR10(root, train=False, download=True, transform=t))


def prepare_cifar100(data_dir='./data'):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    root = os.path.join(data_dir, 'cifar100')
    return (datasets.CIFAR100(root, train=True, download=True, transform=t),
            datasets.CIFAR100(root, train=False, download=True, transform=t))


def prepare_tiny_imagenet(data_dir="./data/tiny-imagenet-200/"):
    t_train = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
    ])
    t_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
    ])
    train_set = TinyImageNet(data_dir, train=True,  transform=t_train)
    test_set  = TinyImageNet(data_dir, train=False, transform=t_val)

    # 兼容：给 TinyImageNet 添加 .targets 属性（供 Dirichlet 划分使用）
    for ds in (train_set, test_set):
        if not hasattr(ds, 'targets'):
            if hasattr(ds, 'labels'):
                ds.targets = list(ds.labels)
            else:
                # 通用 fallback：遍历数据集
                ds.targets = [int(y) for _, y in ds]

    return train_set, test_set


# -------------------------
# Splits
# -------------------------
def _targets_array(trainset):
    """兼容 torchvision 标准数据集和自定义数据集"""
    if hasattr(trainset, 'targets'):
        return np.array(trainset.targets, dtype=int)
    if hasattr(trainset, 'labels'):
        return np.array(trainset.labels, dtype=int)
    # 通用 fallback（慢，但可靠）
    return np.array([int(y) for _, y in trainset], dtype=int)


def iid_split(trainset, num_clients, seed=1):
    rng = np.random.RandomState(seed)
    all_idxs = np.arange(len(trainset))
    rng.shuffle(all_idxs)
    return [s.astype(int).tolist() for s in np.array_split(all_idxs, num_clients)]


def _client_class_hist(client_indices, labels, num_classes):
    hist = np.zeros((len(client_indices), num_classes), dtype=int)
    for cid, idxs in enumerate(client_indices):
        if len(idxs) == 0:
            continue
        lbl = labels[np.array(idxs, dtype=int)]
        hist[cid] = np.bincount(lbl, minlength=num_classes)
    return hist


def dirichlet_split(trainset, num_clients, alpha=0.5, seed=1,
                    min_size=10, max_retry=500, min_classes=0):
    """Dirichlet non-IID split（unbalanced），和旧版 separate_data 行为一致"""
    rng    = np.random.RandomState(seed)
    labels = _targets_array(trainset)
    num_classes = int(labels.max() + 1)

    for _ in range(max_retry):
        client_indices = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            rng.shuffle(idx_k)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            cuts = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            for cid, part in enumerate(np.split(idx_k, cuts)):
                client_indices[cid].extend(part.astype(int).tolist())

        sizes = [len(ci) for ci in client_indices]
        if min(sizes) < min_size:
            continue
        if min_classes > 0:
            hist = _client_class_hist(client_indices, labels, num_classes)
            if (hist > 0).sum(axis=1).min() < min_classes:
                continue

        for cid in range(num_clients):
            rng.shuffle(client_indices[cid])
        return client_indices

    raise RuntimeError(
        f"dirichlet_split failed after {max_retry} retries. "
        f"Try larger alpha or smaller min_size/min_classes."
    )


# -------------------------
# 数据集元信息
# -------------------------
_DATASET_META = {
    "cifar10":       {"num_classes": 10,  "num_channels": 3},
    "cifar100":      {"num_classes": 100, "num_channels": 3},
    "tiny_imagenet": {"num_classes": 200, "num_channels": 3},
}


# -------------------------
# Main API（兼容旧版接口）
# -------------------------
def get_dataset(args):
    """
    接口与旧版 get_dataset.py 完全兼容。

    args 必须字段：
        dataset        : str  ('cifar10', 'cifar100', 'TinyImagenet' / 'tiny_imagenet')
        num_users      : int
        iid            : bool
        data_beta      : float  (Dirichlet alpha，仅 non-IID 时使用)
        seed           : int
        generate_data  : bool

    args 可选字段（有默认值）：
        noniid_case    : int   (旧版字段，新版忽略，统一用 Dirichlet)
        min_size       : int   (default 10)
        min_classes    : int   (default 0)
        max_retry      : int   (default 500)

    副作用（和旧版一致）：
        自动设置 args.num_classes 和 args.num_channels
    """
    # 统一数据集名称
    ds_name = _normalize_dataset_name(args.dataset)
    args.dataset = ds_name   # 回写，保持后续代码一致

    if ds_name not in _DATASET_META:
        raise ValueError(
            f"get_dataset_new 只支持 cifar10 / cifar100 / tiny_imagenet，"
            f"收到: {args.dataset}"
        )

    # 设置 num_classes / num_channels（与旧版行为一致）
    meta = _DATASET_META[ds_name]
    args.num_classes  = meta["num_classes"]
    args.num_channels = meta["num_channels"]

    # 加载数据集
    data_dir = getattr(args, "data_dir", "./data")
    if ds_name == "cifar10":
        train_set, test_set = prepare_cifar10(data_dir=data_dir)
    elif ds_name == "cifar100":
        train_set, test_set = prepare_cifar100(data_dir=data_dir)
    else:
        data_dir = getattr(args, "data_dir", "./data/tiny-imagenet-200/")
        train_set, test_set = prepare_tiny_imagenet(data_dir=data_dir)

    # 尝试读取缓存的划分
    record_path = _build_record_path(args)
    if (not bool(getattr(args, "generate_data", False))) and os.path.exists(record_path):
        dict_users, _ = _load_record(record_path)
        return train_set, test_set, dict_users

    # 生成新划分
    seed      = int(args.seed)
    num_users = int(args.num_users)
    alpha     = float(args.data_beta)
    min_size    = int(getattr(args, "min_size",    10))
    min_classes = int(getattr(args, "min_classes",  0))
    max_retry   = int(getattr(args, "max_retry",  500))

    if bool(args.iid):
        client_indices = iid_split(train_set, num_users, seed=seed)
        extra_meta = {"split": "iid", "seed": seed}
    else:
        client_indices = dirichlet_split(
            train_set, num_users, alpha=alpha, seed=seed,
            min_size=min_size, max_retry=max_retry, min_classes=min_classes,
        )
        extra_meta = {
            "split": "dirichlet_unbalanced",
            "alpha": alpha, "seed": seed,
            "min_size": min_size, "min_classes": min_classes,
        }

    dict_users = {i: client_indices[i] for i in range(num_users)}
    _save_record(record_path, args, dict_users, extra_meta=extra_meta)
    return train_set, test_set, dict_users
