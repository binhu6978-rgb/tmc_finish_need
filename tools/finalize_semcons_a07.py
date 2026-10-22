"""Audit and publish completed observational evidence. Never launches training."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_semcons_a07 import auc


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def csv_rows(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def publish(raw, historical, output):
    run = raw/'a07/observe_400_seed1'
    state = read_json(raw/'status.json'); completed = read_json(run/'completed.json')
    summary = read_json(run/'summary.json'); mechanism = read_json(raw/'analysis/mechanism_summary.json')
    manifest = read_json(run/'observation_manifest.json')
    scores = csv_rows(run/'accuracy_each_round.csv'); old = csv_rows(historical/'accuracy_each_round.csv')
    diagnostics = csv_rows(run/'method_diagnostics.csv'); previous = csv_rows(historical/'method_diagnostics.csv')
    observation = lines(run/'observation_rounds.jsonl'); regression = lines(run/'regression.jsonl')
    utilities = lines(run/'utility_rounds.jsonl'); pairs = lines(run/'utility_pairs.jsonl')
    assert state['status']=='completed' and state['exit_code']==0 and not state['formal_gate_started']
    assert completed['success'] and completed['rounds_completed']==completed['requested_epochs']==400
    assert mechanism['status']=='No-Go' and not mechanism['formal_gate_authorized']
    assert not list(raw.rglob('*.pt'))
    for stream in (scores, diagnostics, observation):
        assert [int(r['round']) for r in stream]==list(range(1,401))
    assert [r['round'] for r in regression]==list(range(401))
    assert all(len(r['models'])==7 for r in regression)
    assert all(math.isfinite(float(r[k])) for r in scores for k in ('full_accuracy','full_ce'))
    metrics_identical = all(float(a[k])==float(b[k]) for a,b in zip(scores,old) for k in ('full_accuracy','full_ce'))
    assert metrics_identical and len(old)==400
    current_slots = [json.loads(r['metrics_json'])['slots'] for r in diagnostics]
    old_slots = [json.loads(r['metrics_json'])['slots'] for r in previous]
    assert current_slots==old_slots
    assert regression[0]['models']==read_json(historical/'manifest.json')['initial_state_sha256']
    assert all(r['nonneutral_gate_count']==r['affected_parameter_numel']==r['gate_learning_delta_l2']==0 and r['gamma']==1 for r in observation)
    expected_rounds = sorted(sum(manifest['config']['diagnostic_rounds'].values(), []))
    assert [r['round'] for r in utilities]==expected_rounds
    assert observation[0]['semantic']['updated']==0
    seeds = manifest['config']['head_seeds']
    for r in observation:
        assert r['head_sha256']==observation[-1]['head_sha256']
        for client in r['clients']:
            if client is None:
                continue
            for value in client['reliability'].values():
                assert value['age'] is None or 0<value['age']<=100
            for head in client['heads'].values():
                age = head['evidence']['mean_anchor_age']
                assert age is None or 0<age<=100
    by_pair = {}
    for r in pairs:
        assert r['sender']!=r['receiver'] and r['round'] in expected_rounds
        slots = current_slots[r['round']-1]
        assert slots[r['sender_slot']]==[r['sender'],r['sender_profile']]
        assert slots[r['receiver_slot']]==[r['receiver'],r['receiver_profile']]
        region = json.loads(diagnostics[r['round']-1]['metrics_json'])['server']['regions'][r['region']]
        assert r['sender_profile'] in region['support_profiles'] and r['receiver_profile'] in region['support_profiles']
        alpha = region['native_slot_shares'][region['slots'].index(r['sender_slot'])]
        assert abs(r['alpha']-alpha)<1e-14
        assert r['negative_loo']==(r['direction_cosine'] is not None and r['direction_cosine']<0)
        weighted = sum(c['delta_ce']*c['count'] for c in r['per_class'])/sum(c['count'] for c in r['per_class'])
        assert abs(weighted-r['delta_ce'])<1e-12
        by_pair.setdefault(r['pair_id'], []).append(r)
    assert len(by_pair)==3996 and all(sorted(r['head_seed'] for r in v)==seeds for v in by_pair.values())
    for v in by_pair.values():
        assert all((r['delta_ce'],r['per_class'])==(v[0]['delta_ce'],v[0]['per_class']) for r in v)
    primary = [v[0] for v in by_pair.values()]
    valid = [r for r in primary if r['split']=='confirmation']
    for name, result in mechanism['scores'].items():
        selected = [r for r in valid if r['evidence'][name] is not None]
        assert abs(auc([r['evidence'][name] for r in selected],[r['delta_ce']>0 for r in selected])-result['confirmation']['auc'])<1e-14
    values = [float(r['full_accuracy']) for r in scores]; peak = max(scores,key=lambda r:float(r['full_accuracy']))
    assert float(peak['full_accuracy'])==summary['best_full_accuracy'] and int(peak['round'])==summary['best_round']
    assert abs(statistics.mean(values[-20:])-summary['last20_full_mean'])<1e-12
    assert abs(statistics.mean(values[-50:])-summary['last50_full_mean'])<1e-12
    finite = read_json(run/'final_model_check.json')
    assert finite['finite_full_parameters'] and finite['full_numel']==11424356
    output.mkdir(parents=True, exist_ok=False)
    for name in ('accuracy_each_round.csv','accuracy_summary.csv','summary.json','completed.json','manifest.json',
                 'observation_manifest.json','resources.json','final_model_check.json','utility_rounds.jsonl'):
        shutil.copyfile(run/name, output/name)
    shutil.copyfile(raw/'status.json',output/'status.json')
    shutil.copytree(raw/'analysis',output/'server_analysis')
    index = []
    for path in [run/'utility_pairs.jsonl',run/'observation_rounds.jsonl',run/'method_diagnostics.csv',run/'regression.jsonl',raw/'observation_400.log']:
        content = path.read_bytes(); target = output/(path.name+'.gz')
        with target.open('wb') as f:
            with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as g:
                g.write(content)
        index.append(dict(file=target.name,raw_file=path.name,raw_bytes=len(content),
            raw_sha256=hashlib.sha256(content).hexdigest(),gzip_bytes=target.stat().st_size,
            gzip_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    write_json(output/'raw_scalar_index.json',index)
    subgroups = {}
    for split in ('all','development','confirmation'):
        subgroups[split] = {}
        for negative in (False,True):
            part = [r for r in primary if (split=='all' or r['split']==split) and r['negative_loo']==negative]
            subgroups[split][str(negative)] = dict(pairs=len(part),harmful=sum(r['delta_ce']>0 for r in part),
                beneficial=sum(r['delta_ce']<0 for r in part),senders=len({r['sender'] for r in part}),receivers=len({r['receiver'] for r in part}))
    readiness = {}
    for stage, subset in [('early',observation[:40]),('middle',observation[179:210]),('late',observation[369:])]:
        clients = [c for r in subset for c in r['clients'] if c]
        readiness[stage] = dict(slots=len(clients),missing_score=sum(c['heads']['17']['evidence']['S0'] is None for c in clients),
            client_ready=sum(c['reliability']['client']['ready'] for c in clients),
            client_profile_ready=sum(c['reliability']['client_profile']['ready'] for c in clients))
    support_shift = {}
    for key in ('client','client_profile'):
        dev = [r['reliability'][key]['support'] for r in primary if r['split']=='development' and r['evidence']['S0'] is not None]
        confirm = [r['reliability'][key]['support'] for r in primary if r['split']=='confirmation' and r['evidence']['S0'] is not None]
        support_shift[key] = dict(development_range=[min(dev),max(dev)],confirmation_range=[min(confirm),max(confirm)],
            confirmation_above_development_fraction=sum(v>max(dev) for v in confirm)/len(confirm))
    audit = dict(decision='No-Go',source_commit=manifest['commit'],config_sha256=manifest['config']['config_sha256'],
        partition_sha256=manifest['partition_sha256'],actual_rounds=400,physical_pairs=3996,head_records=len(pairs),
        all400_full_accuracy_ce_identical_to_frozen_a07=metrics_identical,all400_schedule_identical=True,
        all7_initial_states_identical=True,exact400_historical_weights_comparison='Unavailable: historical tensors were not saved; exact all-state evidence is the separate 2/20 regression',
        head_hashes=observation[-1]['head_sha256'],no_pt=True,nonneutral_gates=0,semantic_updates=sum(r['semantic']['updated'] for r in observation),
        final_anchor_classes=observation[-1]['semantic']['classes_by_channel'],subgroups=subgroups,readiness=readiness,support_shift=support_shift,
        utility_seconds=sum(r['seconds'] for r in utilities),utility_forward_examples=sum(r['receiver_forward_examples'] for r in utilities),
        server_seconds_including_utility=sum(float(r['server_seconds']) for r in scores),evaluation_seconds=sum(float(r['evaluation_seconds']) for r in scores))
    write_json(output/'audit_summary.json',audit)
    plots(scores, old, valid, mechanism, output)
    return audit


def plots(scores, old, pairs, mechanism, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for name in ('S0','S1','S2'):
        part=[r for r in pairs if r['evidence'][name] is not None]
        order=sorted(part,key=lambda r:r['evidence'][name],reverse=True)
        positives=sum(r['delta_ce']>0 for r in order); negatives=len(order)-positives
        yy=np.array([r['delta_ce']>0 for r in order]); change=np.r_[np.flatnonzero(np.diff([r['evidence'][name] for r in order])),len(order)-1]
        axes[0].plot(np.r_[0,np.cumsum(~yy)[change]/negatives,1],np.r_[0,np.cumsum(yy)[change]/positives,1],label=f'{name} AUC={mechanism["scores"][name]["confirmation"]["auc"]:.3f}')
    axes[0].plot([0,1],[0,1],':',color='gray');axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Held-out recipient harm (652 pairs)');axes[0].legend(fontsize=8)
    for i,(seed,value) in enumerate(mechanism['head_sensitivity'].items()):
        a=value['confirmation']['auc']; bounds=value['bootstrap']['sender']['auc_ci95']
        axes[1].errorbar(i,a,yerr=[[a-bounds[0]],[bounds[1]-a]],fmt='o',capsize=4,color='C0')
    axes[1].axhline(.5,color='gray',linestyle=':');axes[1].axhline(.6,color='red',linestyle='--',label='Precommitted minimum')
    axes[1].set(xticks=range(3),xticklabels=['17','29','43'],xlabel='Frozen head seed (not FL seeds)',ylabel='AUC with sender-cluster 95% interval',title='Development-selected S0');axes[1].legend(fontsize=8)
    fig.savefig(output/'utility_evidence.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
    for axis,key,title in zip(axes,('full_accuracy','full_ce'),('Full accuracy (%)','Full CE')):
        axis.plot(range(1,401),[float(r[key]) for r in old],label='Frozen A07',linewidth=1.8)
        axis.plot(range(1,401),[float(r[key]) for r in scores],'--',label='A07 + observation (identical)',linewidth=1)
        axis.set(xlabel='Communication round',ylabel=title);axis.legend(fontsize=8)
    fig.savefig(output/'a07_regression_curve.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('raw',type=Path);p.add_argument('--historical',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    opt=p.parse_args();print(json.dumps(publish(opt.raw,opt.historical,opt.output),indent=2))
