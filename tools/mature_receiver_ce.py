"""One bounded read-only mature-state diagnostic; never resumes global training."""
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapterfl_methods import isolated_evaluation_rng, rng_state
from models.Fed import get_model_list, split_model, select_clients
from models.Update import LocalUpdate_AdaptiveFL
from semcons_scope import ScopeHooks
from tools.receiver_ce_probe import ReceiverProbe
from utils.set_seed import set_random_seed

PROTOCOL_PATH = ROOT / 'configs/mature_q_final_probe.json'
CORE = ('models/Fed.py', 'models/Update.py', 'models/resnet.py',
        'utils/get_datatset_new.py', 'utils/set_seed.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for k, v in sorted(state.items()):
        v = v.detach().cpu().contiguous()
        digest.update((k + str(tuple(v.shape)) + str(v.dtype)).encode())
        digest.update(v.numpy().tobytes())
    return digest.hexdigest()


def rng_sha():
    # Pickle tensor objects contains storage identity; hash values instead.
    s = rng_state()
    return hashlib.sha256(pickle.dumps((s['python'], s['numpy'])) + s['torch'].cpu().numpy().tobytes()
                          + b''.join(x.cpu().numpy().tobytes() for x in s['cuda'])).hexdigest()


class TrainingOnlyCIFAR100(Dataset):
    """Exact native CIFAR100 training transform, without opening any test file."""
    def __init__(self, root):
        path = Path(root) / 'cifar-100-python' / 'train'
        content = path.read_bytes()
        if hashlib.md5(content).hexdigest() != '16019d7e3df5f24257cddd939b257f8d':
            raise ValueError('CIFAR100 train checksum differs')
        entry = pickle.loads(content, encoding='latin1')
        self.data = entry['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.targets = entry['fine_labels']
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.data[i])), self.targets[i]


def fixed_schedule(args, protocol):
    with isolated_evaluation_rng():
        random.seed(protocol['seed']); np.random.seed(protocol['seed'])
        schedule = []
        for c in range(protocol['cohorts']):
            profiles = np.random.choice(range(7), protocol['slots_per_cohort'])
            clients = select_clients(args, profiles, 7)
            schedule.append({'cohort': c + 1, 'profiles': profiles.tolist(), 'clients': clients})
        return schedule


def restore_profiles(models, anchor):
    """Use original prefix restoration; load_state_dict copies, no shared storage."""
    hashes = []
    for net in models:
        expected = split_model(anchor, net.state_dict())
        net.load_state_dict(expected, strict=True)
        actual = net.state_dict()
        if not all(actual[k].shape == expected[k].shape and torch.equal(actual[k].cpu(), expected[k].cpu()) for k in actual):
            raise AssertionError('Seven-profile prefix restoration differs')
        hashes.append(state_sha(actual))
    return hashes


def verify_checkpoint(checkpoint, protocol, partition):
    ident = checkpoint['identity']
    required = dict(seed=1, epochs=400, num_users=100, frac=.1, local_ep=5, local_bs=50,
                    lr=.01, lr_decay=.998, momentum=.5, weight_decay=.0001, optimizer='sgd',
                    dataset='cifar100', model='resnet', iid=0, data_beta=.3,
                    client_hetero_ration='4:3:3', client_chosen_mode='available', method='m2')
    if any(ident.get(k) != v for k, v in required.items()):
        raise ValueError('E03 training identity differs')
    if checkpoint['next_round'] != 400 or len(checkpoint['rows']) != 400 or ident['options']['conflict_lambda'] != 4:
        raise ValueError('Not completed E03 final400')
    if sha(partition) != protocol['partition_sha256'] or ident['partition_sha256'] != sha(partition):
        raise ValueError('Original partition differs')
    if any(sha(ROOT / name) != ident['source_sha256'][name] for name in CORE):
        raise ValueError('Original model/local-SGD/split/data source differs from E03')
    if not all(torch.isfinite(v).all() for v in checkpoint['global'].values()):
        raise ValueError('Nonfinite mature checkpoint')


def execute(opt):
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if sha(opt.checkpoint) != protocol['checkpoint_sha256']:
        raise ValueError('Only the predeclared E03 latest400 checksum is allowed')
    checkpoint = torch.load(opt.checkpoint, map_location='cpu', weights_only=False)
    verify_checkpoint(checkpoint, protocol, opt.partition)
    if sha(opt.manifest) != protocol['e03_manifest_sha256']:
        raise ValueError('E03 original training manifest differs')
    saved_manifest = json.loads(opt.manifest.read_text())
    if saved_manifest['identity'] != checkpoint['identity']:
        raise ValueError('Manifest/checkpoint identity differs')
    config = saved_manifest['training_configuration']
    args = SimpleNamespace(**config)
    args.device = torch.device('cuda:0'); args.method = 's07'; args.conflict_lambda = 4.
    # Observer only: no S07 aggregate/history update is called.
    schedule = fixed_schedule(args, protocol)
    root = opt.output_root
    root.mkdir(parents=True, exist_ok=False)
    frozen = dict(protocol, schedule=schedule, protocol_sha256=sha(PROTOCOL_PATH))
    (root / 'protocol.json').write_text(json.dumps(frozen, indent=2))
    data = TrainingOnlyCIFAR100(opt.dataset_root)
    saved = json.loads(opt.partition.read_text())
    users = {int(k): v for k, v in saved['train_data'].items()}
    all_indices = [i for v in users.values() for i in v]
    if len(data) != 50000 or len(users) != 100 or len(all_indices) != 50000 or set(all_indices) != set(range(50000)):
        raise ValueError('Full original training partition required')
    torch.set_num_threads(2)
    set_random_seed(1)
    anchor = checkpoint['global']; anchor_hash = state_sha(anchor)
    models, info = get_model_list(args)
    restored = restore_profiles(models, anchor)
    hook = ScopeHooks(args, models)
    with isolated_evaluation_rng():
        observer = ReceiverProbe(data, users, root)
    observer.templates = models
    resource = {'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
                'python': sys.version, 'cuda': torch.version.cuda, 'peak_allocated_bytes': 0}
    manifest = dict(checkpoint_sha256=sha(opt.checkpoint), checkpoint_path=str(opt.checkpoint.resolve()),
                    checkpoint_round=400, restored_profile_sha256=restored, profile_info=info,
                    anchor_sha256=anchor_hash, checkpoint_identity=checkpoint['identity'],
                    training_configuration=config, data_root=str(opt.dataset_root.resolve()),
                    partition_sha256=sha(opt.partition), train_examples=len(data),
                    official_test_loaded=False, source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    source_sha256={name: sha(ROOT/name) for name in (*CORE, 'semcons_scope.py',
                        'tools/receiver_ce_probe.py','tools/mature_receiver_ce.py','tools/analyze_mature_q.py')},
                    warning='Independent E03 final-state samples; not S07 natural late training. No aggregate or Gate feedback.')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    set_random_seed(1)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        for item in schedule:
            ci = item['cohort'] - 1
            before_hashes = restore_profiles(models, anchor)
            if before_hashes != restored: raise AssertionError('Mature dispatch differs across cohorts')
            hook.start_round(ci)
            uploads = []; dispatched = []; lens = []; train_seconds = 0.; metrics = []
            for cid, profile in zip(item['clients'], item['profiles']):
                net = copy.deepcopy(models[profile])
                sent = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                r = rng_sha(); before = hook.before_slot(net, data, users[cid], cid)
                if rng_sha() != r: raise AssertionError('Pre-measurement consumed training RNG')
                local = LocalUpdate_AdaptiveFL(args, data, users[cid])
                t = time.perf_counter()
                trained = local.train(round=protocol['sgd_round_index'], net=net)
                train_seconds += time.perf_counter() - t
                if not all(torch.isfinite(v).all() for v in trained.values()): raise FloatingPointError('Nonfinite upload')
                r = rng_sha(); metric = hook.finish_slot(net, data, before, cid, profile)
                if rng_sha() != r: raise AssertionError('Post-measurement consumed training RNG')
                metrics.append(metric)
                uploads.append({k:v.detach().cpu().clone() for k,v in trained.items()})
                dispatched.append(sent); lens.append(len(users[cid]))
                del net, local, trained
            r = rng_sha()
            with isolated_evaluation_rng(): result = observer.measure(hook, uploads, lens, dispatched)
            unchanged = (rng_sha() == r and state_sha(anchor) == anchor_hash
                         and [state_sha(m.state_dict()) for m in models] == restored and hook.history == {})
            if not unchanged: raise AssertionError('Read-only diagnostic modified anchor/templates/RNG/history')
            result.update(independent_mature_cohort=True, sgd_round_index=400, native_train_seconds=train_seconds,
                          local_metrics=metrics, anchor_rng_history_unchanged=True,
                          dispatched_sha256=[state_sha(s) for s in dispatched], upload_sha256=[state_sha(s) for s in uploads])
            with (root / 'receiver_ce.jsonl').open('a') as f: f.write(json.dumps(result) + '\n')
            del uploads, dispatched
        resource['status'] = 'completed_fixed_budget'
        (root / 'diagnostic_completed.json').write_text(json.dumps({'cohorts':5,'slots':50,'success':True,
            'global_updates':0,'official_test_evaluations':0,'anchor_sha256':anchor_hash,'formal400':False},indent=2))
    except BaseException:
        resource['status'] = 'failed_no_retry'
        raise
    finally:
        resource.update(wall_seconds=time.perf_counter()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
        (root / 'resources.json').write_text(json.dumps(resource, indent=2))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--partition', type=Path, default=ROOT/'data/cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json')
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    execute(p.parse_args())


if __name__ == '__main__': main()
