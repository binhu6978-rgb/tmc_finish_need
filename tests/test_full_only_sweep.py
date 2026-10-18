import json
from pathlib import Path
import subprocess
import sys
import pytest
import tools.run_full_only_sweep as sweep
from utils.options import args_parser


def test_fixed_ten_commands_and_hooks(tmp_path,monkeypatch):
    from test_adapterfl_methods import tiny_models
    from adapterfl_methods import MethodHooks
    batches=sweep.plan(tmp_path,[0,0],sys.executable)
    assert [[j['id'] for j in pair] for pair in batches]==[['B00','E02'],['E01','E03'],['E04','E05'],['E06','E07'],['E08','E09']]
    expected={'E01':('conflict_lambda',.5),'E02':('conflict_lambda',2.),'E03':('conflict_lambda',4.),
              'E04':('partition_alpha_remainder',.95),'E05':('partition_alpha_remainder',1.05),
              'E06':('partition_alpha_shared',1.05),'E07':('shared_prox_mu',.003),
              'E08':('shared_prox_mu',.03),'E09':('history_beta',.5)}
    paths=[]
    for job in [j for pair in batches for j in pair]:
        monkeypatch.setattr(sys,'argv',['test',*job['argv'][3:]])
        args=args_parser();hooks=MethodHooks(args,tiny_models())
        assert args.epochs==400 and args.eval_profiles=='full' and args.seed==1 and args.gpu==0
        assert 'full_only_400' in args.method_run_name
        if job['id'] in expected:
            key,value=expected[job['id']];assert hooks.config[key]==value
        paths.append(job['output_dir'])
    assert len(set(paths))==10
    assert not tmp_path.joinpath('experiments').exists()


def test_dry_run_has_no_outputs_and_external_gpu_is_preserved(tmp_path,capsys):
    sweep.main(['--dry-run','--gpus','0,0','--output-root',str(tmp_path/'unused')])
    assert not (tmp_path/'unused').exists()
    text=capsys.readouterr().out
    assert text.count('CUDA_VISIBLE_DEVICES=0 ')==10 and '--eval-profiles full' in text
    # Execute only the entry point's environment preamble: no Torch import.
    source=(sweep.ROOT/'main_fed_ori.py').read_text(encoding='utf-8').split('import matplotlib')[0]
    import os
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='1'
    assert subprocess.check_output([sys.executable,'-c',source+'\nprint(os.environ["CUDA_VISIBLE_DEVICES"])'],env=env).strip()==b'1'


def test_batch_failure_stops_own_peer_and_keeps_error(tmp_path):
    jobs=[]
    for name,code in [('fail',"import sys;print('ORIGINAL_ERROR',flush=True);sys.exit(7)"),('peer','import time;time.sleep(60)')]:
        jobs.append({'id':name,'gpu':0,'argv':[sys.executable,'-u','-c',code],'stdout':str(tmp_path/(name+'.log'))})
    state={'jobs':{}}
    with pytest.raises(RuntimeError,match='exited 7'):
        sweep.run_batch(jobs,state,lambda:None,validator=lambda j:None)
    assert 'ORIGINAL_ERROR' in (tmp_path/'fail.log').read_text()
    assert state['jobs']['fail']['returncode']==7
    assert state['jobs']['peer']['state']=='stopped_after_batch_failure'


def test_shared_gpu_override_and_existing_directory_protection(tmp_path):
    with pytest.raises(SystemExit):
        sweep.main(['--execute','--gpus','0,0','--output-root',str(tmp_path/'out')])
    with pytest.raises(FileExistsError):
        sweep.main(['--execute','--gpus','0,0','--allow-shared-gpu','--output-root',str(tmp_path)])


def test_gpu_mapping_and_unknown_process_protection(monkeypatch):
    def check(argv,**kw):
        if len(argv)==1:return 'nvidia inventory'
        if '--query-gpu=index,uuid,memory.free' in argv:return '0, UUID0, 22000\n1, UUID1, 23000\n'
        return 'UUID1, 1234\n'
    monkeypatch.setattr(sweep.subprocess,'check_output',check)
    assert sweep.gpu_inventory([0,0])=='nvidia inventory'
    with pytest.raises(RuntimeError,match='another compute process'):sweep.gpu_inventory([0,1])


def test_prior_queue_failure_or_completion(tmp_path):
    path=tmp_path/'prior.json'
    path.write_text(json.dumps({'state':'failed'}))
    with pytest.raises(RuntimeError,match='Prior queue failed'):sweep.wait_for_prior(path,{},lambda:None)
    path.write_text(json.dumps({'state':'completed','jobs':{str(i):{'state':'completed'} for i in range(7)}}))
    sweep.wait_for_prior(path,{},lambda:None)
