"""Prepare fixed training partitions before any of the forty processes start."""
import hashlib
import json
from pathlib import Path

from torchvision import datasets, transforms
import utils.get_datatset_new as original

ORIGINAL_SHA = '41ab2d3486663ce0100e183ab00565de137f91af07632c236c2ddf60d890ba41'
ORIGINAL_RECORD = Path('data/cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json')


def config():
    return json.loads(Path('configs/capability_final40_400.json').read_text())


def datasets_for(condition, root, download=False):
    """Use the project's actual transforms, including CIFAR10 train flip."""
    if condition['dataset'] == 'cifar100':
        mean, std = (.5071, .4867, .4408), (.2675, .2565, .2761)
        cls, operations = datasets.CIFAR100, []
    else:
        mean, std = (.4914, .4822, .4465), (.2023, .1994, .2010)
        cls, operations = datasets.CIFAR10, [transforms.RandomHorizontalFlip()]
    end = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    train = cls(str(root), train=True, download=download, transform=transforms.Compose(operations + end))
    test = cls(str(root), train=False, download=download, transform=transforms.Compose(end))
    if len(train) != 50000 or len(test) != 10000:
        raise ValueError('Full real CIFAR data required')
    if set(train.targets) != set(range(condition['classes'])) or set(test.targets) != set(range(condition['classes'])):
        raise ValueError('Actual dataset labels/classes differ')
    return train, test


def validate_partition(payload, condition):
    if (payload['dataset'] != condition['dataset'] or payload['num_users'] != 100
            or payload['seed'] != 1 or payload['data_beta'] != condition['alpha'] or payload['iid']):
        raise ValueError('Prepared partition condition mismatch')
    users = {int(k): v for k, v in payload['train_data'].items()}
    if set(users) != set(range(100)) or min(map(len, users.values())) < 10:
        raise ValueError('Invalid physical client partition')
    if sorted(i for indices in users.values() for i in indices) != list(range(50000)):
        raise ValueError('Partition must cover all training samples once')
    return users


def context_for(name, condition, users):
    ratio = [float(x) for x in condition['ratio'].split(':')]
    proportions = [round(x / sum(ratio), 2) for x in ratio]
    boundaries = [int(100 * proportions[0]), int(100 * sum(proportions[:2]))]
    population = [boundaries[0], boundaries[1] - boundaries[0], 100 - boundaries[1]]
    edges = [0, *boundaries, 100]
    return dict(condition=name, population=population, boundaries=boundaries,
        data_mass=[sum(len(users[i]) for i in range(a, b)) for a, b in zip(edges, edges[1:])],
        partition_rule='original dirichlet_split_unbalanced, seed1/min_size10/max_retry500; original alpha.3 retained verbatim')


def prepare(dataset_root, output):
    output.mkdir(parents=True, exist_ok=False)
    # All random draws are in the original splitter's local RandomState(1).
    # This is a separate pre-training process, never a training RNG stream.
    manifest = {}
    cache = {}
    for name, condition in config()['conditions'].items():
        train, _ = datasets_for(condition, dataset_root, download=True)
        key = condition['dataset'], condition['alpha']
        if key not in cache:
            if key == ('cifar100', .3):
                raw = ORIGINAL_RECORD.read_bytes()
                if hashlib.sha256(raw).hexdigest() != ORIGINAL_SHA:
                    raise ValueError('Original A/S4 reference partition differs')
                payload = json.loads(raw)
            else:
                split = original.dirichlet_split_unbalanced(train, 100, alpha=condition['alpha'], seed=1,
                                                           min_size=10, max_retry=500, min_classes=0)
                payload = dict(dataset=condition['dataset'], num_users=100, iid=False,
                    data_beta=condition['alpha'], seed=1, train_data={i: s for i, s in enumerate(split)},
                    extra_meta=dict(split='dirichlet_unbalanced', alpha=condition['alpha'], seed=1,
                                    min_size=10, min_classes=0, max_retry=500))
                raw = json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8')
            cache[key] = raw
        raw = cache[key]
        users = validate_partition(json.loads(raw), condition)
        target = output / (name + '.json')
        target.write_bytes(raw)
        manifest[name] = dict(condition=condition, path=str(target),
            sha256=hashlib.sha256(raw).hexdigest(), context=context_for(name, condition, users),
            train_examples=50000, test_examples=10000)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest
