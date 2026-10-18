"""Standard-library-only archive integrity check."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root=Path(root)
    m=json.loads((root/'BUNDLE_MANIFEST.json').read_text())
    for name,digest in m['sha256'].items():
        p=root/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Package content differs: {name}')
    print(f"Package verified: {len(m['sha256'])} files")


if __name__=='__main__': verify(Path(__file__).resolve().parents[1])
