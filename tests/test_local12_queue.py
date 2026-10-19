import json
import os
from pathlib import Path

import pytest

from tools.run_local12 import ROOT,environment,jobs,run


def test_fixed_twelve_commands_paths_batches_and_ref():
    config=json.loads((ROOT/'configs/e03_local12_400.json').read_text())
    batches=jobs(config,Path('python'),Path('dataset'),Path('output'))
    assert len(batches)==6 and all(len(b)==2 for b in batches)
    flat=sum(batches,[])
    assert [j['id'] for j in flat]==['REF']+[f'L{i:02d}' for i in range(1,12)]
    assert len({j['output'] for j in flat})==12
    assert flat[0]['method']=='m2' and '--local-reference' in flat[0]['argv']
    for j in flat:
        assert j['argv'][j['argv'].index('--epochs')+1]=='400'
        assert j['argv'][j['argv'].index('--eval-profiles')+1]=='full'
    assert environment(Path('python'))['CUDA_VISIBLE_DEVICES']=='0'


def test_queue_no_overwrite(tmp_path):
    with pytest.raises(FileExistsError):run([],tmp_path,os.environ.copy())


def test_two_children_failure_waits_for_peer_and_blocks_later_batch(tmp_path,monkeypatch):
    import tools.report_tonight10 as reporting
    import tools.run_local12 as q
    events=[]
    class Process:
        def __init__(self,argv,**kw):
            self.name=argv[0];self.pid=len(events)+100;events.append(('start',self.name))
        def wait(self):events.append(('wait',self.name));return 1 if self.name=='bad' else 0
    monkeypatch.setattr(q.subprocess,'Popen',Process)
    monkeypatch.setattr(q.subprocess,'check_output',lambda cmd,**kw:'12000' if '--query-gpu=memory.free' in cmd else 'GPU')
    monkeypatch.setattr(reporting,'validate',lambda path:None)
    def job(name):return {'id':name,'argv':[name],'output':str(tmp_path/name)}
    out=tmp_path/'suite'
    assert run([[job('bad'),job('peer')],[job('never'),job('never2')]],out,os.environ.copy())==1
    assert events==[('start','bad'),('start','peer'),('wait','bad'),('wait','peer')]
    state=json.loads((out/'status.json').read_text())
    assert state['state']=='failed' and state['completed']==['peer'] and state['failed']==['bad']


def test_low_memory_changes_only_scheduling(tmp_path,monkeypatch):
    import tools.report_tonight10 as reporting
    import tools.report_local12 as report
    import tools.run_local12 as q
    events=[]
    class Process:
        def __init__(self,argv,**kw):self.name=argv[0];self.pid=1;events.append(('start',self.name))
        def wait(self):events.append(('wait',self.name));return 0
    monkeypatch.setattr(q.subprocess,'Popen',Process)
    monkeypatch.setattr(q.subprocess,'check_output',lambda cmd,**kw:'4000' if '--query-gpu=memory.free' in cmd else 'GPU')
    monkeypatch.setattr(reporting,'validate',lambda path:None);monkeypatch.setattr(report,'report',lambda path:None)
    batch=[{'id':n,'argv':[n],'output':str(tmp_path/n)} for n in ('a','b')]
    assert run([batch],tmp_path/'suite',os.environ.copy())==0
    assert events==[('start','a'),('wait','a'),('start','b'),('wait','b')]
