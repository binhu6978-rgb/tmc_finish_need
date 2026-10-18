"""Preserve the local tracked source tree in the Linux server archive."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from experiments.run_hgkr7 import CODE_FILES

ARCHIVE_ROOT = 'AAAITMC_finish_need_code'


def is_project_source(name):
    path=Path(name)
    if any(part in {'__pycache__','.git','.vscode','.codex','.agents','runs','tmp'} for part in path.parts):
        return False
    if path.name in {'.gitignore','.gitattributes'}: return True
    return path.suffix.lower() in {'.py','.json','.md','.sh','.txt','.toml','.yaml','.yml','.ini','.cfg'}


def project_files():
    # Git tracks the project sources; it intentionally excludes local/untracked
    # results, credentials and attachments. Old tracked bytecode is filtered too.
    names=subprocess.check_output(['git','ls-files','-z']).decode('utf-8').split('\0')
    return sorted(name for name in names if name and is_project_source(name))


def build(destination):
    paths=project_files()
    missing=set(CODE_FILES)-set(paths)
    if missing: raise ValueError(f'Missing tracked runtime sources: {sorted(missing)}')
    contents={p:Path(p).read_bytes() for p in paths}
    if b'\r' in contents['run_7tier_300.sh']: raise ValueError('Shell script must use LF endings')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    # Capture source bytes independently, so archive integrity never depends on a .git checkout.
    version={'git_commit':commit,'branch':'feature/hgkr-fl-seven-tier-300',
             'protocol':'HGKR7-300-full-eval-v2','source_sha256':{p:hashlib.sha256(contents[p]).hexdigest() for p in CODE_FILES}}
    contents['VERSION.json']=json.dumps(version,indent=2).encode()
    manifest={'sha256':{p:hashlib.sha256(b).hexdigest() for p,b in contents.items()},
              'layout':'local_tracked_source_tree','archive_root':ARCHIVE_ROOT,
              'formal_results_included':False,'CIFAR_raw_data_included':False}
    contents['BUNDLE_MANIFEST.json']=json.dumps(manifest,indent=2).encode()
    dest=Path(destination); dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): raise FileExistsError('Refusing to overwrite an existing package')
    with tarfile.open(dest,'w:gz') as archive:
        for name,value in sorted(contents.items()):
            item=tarfile.TarInfo(ARCHIVE_ROOT+'/'+name); item.size=len(value)
            item.mode=0o755 if name.endswith('.sh') else 0o644
            item.mtime=0
            archive.addfile(item,io.BytesIO(value))
    digest=hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.with_suffix(dest.suffix+'.sha256').write_text(f'{digest}  {dest.name}\n',encoding='ascii')
    print(json.dumps({'archive':str(dest.resolve()),'sha256':digest,'files':len(contents),'bytes':dest.stat().st_size}))
    return dest


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',default='runs/hgkr7_gpu0_delivery/hgkr7_300_full_project_gpu0_server.tar.gz')
    build(p.parse_args().output)
