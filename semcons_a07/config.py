import hashlib
import json
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / 'configs/semcons_a07_observation_v2.json'


def load(path=DEFAULT):
    raw = Path(path).read_bytes()
    cfg = json.loads(raw)
    if cfg['gamma'] != 1 or cfg['formal_gate_enabled']:
        raise ValueError('This frozen observational entry cannot activate semantic gates')
    cfg['config_sha256'] = hashlib.sha256(raw).hexdigest()
    return cfg


def seed_of(*items):
    return int.from_bytes(hashlib.sha256(repr(items).encode()).digest()[:8], 'little') % (2**32)
