"""Small allow-listed Linux bundle: no datasets, runs, Git metadata or private papers."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from experiments.run_hgkr7 import CODE_FILES, PARTITION


def build(destination):
    extra=['run_7tier_300.sh','requirements-hgkr7.txt','README_HGKR7.md',
           'tools/verify_hgkr7_package.py','tests/test_hgkr7.py',
           'utils/__init__.py','optimizer/__init__.py']
    paths=sorted(set(CODE_FILES+[PARTITION]+extra))
    contents={p:Path(p).read_bytes() for p in paths}
    if b'\r' in contents['run_7tier_300.sh']: raise ValueError('Shell script must use LF endings')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    # Capture source bytes independently, so archive integrity never depends on a .git checkout.
    version={'git_commit':commit,'branch':'feature/hgkr-fl-seven-tier-300',
             'protocol':'HGKR7-300-v1','source_sha256':{p:hashlib.sha256(contents[p]).hexdigest() for p in CODE_FILES}}
    contents['VERSION.json']=json.dumps(version,indent=2).encode()
    manifest={'sha256':{p:hashlib.sha256(b).hexdigest() for p,b in contents.items()},
              'formal_results_included':False,'CIFAR_raw_data_included':False}
    contents['BUNDLE_MANIFEST.json']=json.dumps(manifest,indent=2).encode()
    dest=Path(destination); dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): raise FileExistsError('Refusing to overwrite an existing package')
    with tarfile.open(dest,'w:gz') as archive:
        for name,value in sorted(contents.items()):
            item=tarfile.TarInfo('hgkr7_300/'+name); item.size=len(value)
            item.mode=0o755 if name.endswith('.sh') else 0o644
            item.mtime=0
            archive.addfile(item,io.BytesIO(value))
    digest=hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.with_suffix(dest.suffix+'.sha256').write_text(f'{digest}  {dest.name}\n',encoding='ascii')
    print(json.dumps({'archive':str(dest.resolve()),'sha256':digest,'files':len(contents),'bytes':dest.stat().st_size}))
    return dest


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',default='runs/hgkr7_delivery/hgkr7_300_server.tar.gz')
    build(p.parse_args().output)
