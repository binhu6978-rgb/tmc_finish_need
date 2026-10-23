"""Read trusted local checkpoints and journals only; no SGD or restoration run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import torch


def verify(root,historical=None):
    result={}
    for mode in ('neutral','active'):
        path=root/mode/'training/a07'/f'{mode}_500_seed1'
        state=torch.load(path/'recovery_latest.pt',map_location='cpu',weights_only=False)
        manifest=json.loads((root/mode/'manifest.json').read_text())
        assert state['identity']==manifest['identity'] and state['next_round']==500 and len(state['rows'])==500
        h=hashlib.sha256()
        for k,v in state['full'].items():
            assert bool(torch.isfinite(v).all());h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
        last=json.loads((path/'regression.jsonl').read_text().splitlines()[-1])
        assert h.hexdigest()==last['models'][-1]
        bank=state['adapter_bank'];adapter_bytes=sum(v.numel()*v.element_size() for b in bank.values() for v in b.values())
        log=json.loads((path/'semantic_v2_rounds.jsonl').read_text().splitlines()[-1])
        assert adapter_bytes==log['adapter_storage_bytes']
        assert max(v['round'] for v in state['semantic']['tiers'].values())<=500
        assert max(v['round'] for history in state['reliability']['tiers'].values() for v in history)<=500
        result[mode]=dict(checkpoint_round=500,finite_full=True,full_sha256=h.hexdigest(),
            final_journal_state_equal=True,identity_equal=True,adapter_entries=len(bank),adapter_bytes=adapter_bytes,
            semantic_tier_classes=len(state['semantic']['tiers']),reliability_tier_classes=len(state['reliability']['tiers']),
            rng_streams=sorted(state['rng']),file_bytes=(path/'recovery_latest.pt').stat().st_size,
            note='Complete saved state verified; no real resumed training was performed')
    if historical:
        old=list(csv.DictReader(historical.open()));new=list(csv.DictReader((root/'neutral/training/a07/neutral_500_seed1/accuracy_each_round.csv').open()))
        assert len(old)==400 and all(old[i][k]==new[i][k] for i in range(400) for k in ('full_accuracy','full_ce'))
        result['historical_a07_first400']=dict(rounds=400,accuracy_ce_exact=True,
            path=str(historical),sha256=hashlib.sha256(historical.read_bytes()).hexdigest())
    (root/'recovery_and_historical_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);p.add_argument('--historical',type=Path)
    args=p.parse_args();verify(args.root,args.historical)
