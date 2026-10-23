"""Read original journals, audit completion, archive scalar evidence only; never train."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from semcons_a07_v2.pair500 import verify_pair,sampled_rounds


def collect(root):
    protocol=json.loads((ROOT/'configs/semantic_a07_pair500.json').read_text())
    audit=verify_pair(root,500);paths=[]
    for mode in ('neutral','active'):
        run=root/mode/'training/a07'/f'{mode}_500_seed1'
        records=[json.loads(l) for l in (run/'iv_e_rounds.jsonl').read_text().splitlines()]
        assert [r['round'] for r in records]==sampled_rounds(protocol)
        memory=[json.loads(l) for l in (run/'semantic_v2_rounds.jsonl').read_text().splitlines()]
        assert [r['round'] for r in memory]==list(range(1,501))
        assert all(math.isfinite(v) for r in records for n in r['norms'].values()
            for k,v in n.items() if isinstance(v,(float,int)))
        audit[mode]=dict(rounds=500,diagnostic_rounds=len(records),
            physical_receivers=len({r['receiver'] for x in records for r in x['figure7']}),
            figure7_rows=sum(len(r['figure7']) for r in records),figure8_rows=sum(len(r['figure8']) for r in records),
            recovery=json.loads((run/'recovery_latest.json').read_text()))
        assert audit[mode]['recovery']['round']==500
        for name in ('manifest.json','completed.json','resources.json'):
            paths.append(root/mode/name)
        paths.extend(p for p in run.iterdir() if p.suffix in ('.csv','.json','.jsonl'))
    paths.extend(root/name for name in ('status.json','report.log','neutral_500.log','active_500.log','pytest_linux.log','gpu_storage.txt'))
    paths.append(root/'pilot/pilot_pass.json')
    paths=sorted(set(paths));assert all(p.is_file() for p in paths)
    index={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
    (root/'completion_audit.json').write_text(json.dumps(audit,indent=2))
    (root/'scalar_evidence_index.json').write_text(json.dumps(index,indent=2))
    archive=root/'scalar_evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in [*paths,root/'completion_audit.json',root/'scalar_evidence_index.json']:
            tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
    print(json.dumps(dict(audit=audit,archive=str(archive),bytes=archive.stat().st_size,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);collect(p.parse_args().root)
