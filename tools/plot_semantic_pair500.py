"""Recreate Figure7/8 and full curves from the delivered scalar CSVs, no training."""
import argparse
import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.report_semantic_pair500 import draw


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('evidence',type=Path);root=p.parse_args().evidence
    curves={mode:[{k:float(v) for k,v in r.items()} for r in csv.DictReader((root/mode/'accuracy_each_round.csv').open())]
            for mode in ('neutral','active')}
    draw(root,curves)
