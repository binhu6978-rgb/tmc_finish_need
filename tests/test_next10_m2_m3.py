import csv
import json
import sys
import pytest
import tools.run_full_only_sweep as sweep
import tools.report_next10_m2_m3 as reporting
from utils.options import args_parser

CONFIG=sweep.ROOT/'configs/adapterfl_next10_m2_m3_400.json'


def test_next10_plan_and_method_injection(tmp_path,monkeypatch):
    from test_adapterfl_methods import tiny_models
    from adapterfl_methods import MethodHooks
    batches=sweep.plan(tmp_path,[0,0],sys.executable,CONFIG)
    assert [[j['id'] for j in b] for b in batches]==[['N03','N04'],['N01','N02'],['N05','N06'],['N07','N08'],['N09','N10']]
    expected=[3.,3.5,4.5,5.,6.,8.]
    m3={'N07':(1.,.92),'N08':(1.,.98),'N09':(.98,.95),'N10':(1.02,.95)}
    paths=[]
    for job in [j for b in batches for j in b]:
        monkeypatch.setattr(sys,'argv',['test',*job['argv'][3:]])
        args=args_parser();hooks=MethodHooks(args,tiny_models())
        assert args.epochs==400 and args.seed==1 and args.eval_profiles=='full' and args.generate_data==0 and args.data_beta==.3
        assert args.gpu==0 and job['gpu']==0
        if job['method']=='m2':assert hooks.config['conflict_lambda']==expected[int(job['id'][1:])-1]
        else:
            assert (hooks.config['partition_alpha_shared'],hooks.config['partition_alpha_remainder'])==m3[job['id']]
        paths.append(job['output_dir'])
    assert len(set(paths))==10
    assert not (tmp_path/'experiments').exists()


def test_next10_dry_run_no_training_or_output(tmp_path,capsys):
    sweep.main(['--config',str(CONFIG),'--dry-run','--gpus','0,0','--output-root',str(tmp_path/'unused')])
    text=capsys.readouterr().out
    assert text.count('CUDA_VISIBLE_DEVICES=0 ')==10
    assert '--conflict-lambda 8.0' in text and '--partition-alpha-shared 1.02' in text
    assert not (tmp_path/'unused').exists()


def write_fake(path,peak_round,peak):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['round','full_accuracy','full_ce']);writer.writeheader()
        for r in range(1,401):
            writer.writerow({'round':r,'full_accuracy':peak if r==peak_round else 20+r*.01,'full_ce':5-r*.005})


def test_next10_offline_report_synthetic_data_only(tmp_path,monkeypatch):
    monkeypatch.setattr(reporting,'ROOT',tmp_path)
    refs=[]
    for name,peak in [('B00',36.40),('E03',37.29)]:
        p=tmp_path/(name+'.csv');write_fake(p,100,peak)
        refs.append({'id':name,'rounds_csv':name+'.csv'})
    original=(tmp_path/'E03.csv').read_bytes()
    out=tmp_path/'report';out.mkdir()
    batches=sweep.plan(tmp_path/'new',[0,0],sys.executable,CONFIG)
    for batch in batches:
        for j in batch:
            peak_round=1 if j['id']=='N01' else 400 if j['id']=='N02' else 200
            write_fake(__import__('pathlib').Path(j['output_dir'])/'accuracy_each_round.csv',peak_round,38.)
    reporting.report_next10(batches,out,{'references':refs,'peak_ce_window_radius':2})
    with (out/'comparison.csv').open() as f:rows={r['id']:r for r in csv.DictReader(f)}
    assert len(rows)==10
    assert float(rows['N03']['peak_minus_E03_pp'])==pytest.approx(.71)
    assert rows['N01']['best_round']=='1' and rows['N01']['peak_ce_window_end']=='3'
    assert rows['N02']['best_round']=='400' and rows['N02']['peak_ce_window_start']=='398'
    assert float(rows['N03']['peak_ce'])==4.
    assert float(rows['N03']['peak_window_ce_mean'])==pytest.approx(4.)
    assert (out/'full_accuracy_comparison.png').read_bytes().startswith(b'\x89PNG')
    assert len(list((out/'accuracy_curves').glob('N*.png')))==10
    with (out/'full_accuracy_all_rounds.csv').open() as f:assert len(list(csv.DictReader(f)))==400
    assert 'test-oracle' in (out/'report.md').read_text()
    assert (tmp_path/'E03.csv').read_bytes()==original
