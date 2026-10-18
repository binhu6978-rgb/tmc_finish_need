"""Strict completion checks and offline summaries; peak is explicitly test-oracle."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from experiments.baseline import content_hash
from hgkr.seven import MODES


def load_suite(root, expected_rounds=300):
    data={}; ref=None
    for mode in MODES:
        p=Path(root)/mode
        if (p/'failed.json').exists(): raise ValueError(f'{mode} has a failure marker')
        m=json.loads((p/'manifest.json').read_text()); done=json.loads((p/'completed.json').read_text())
        rows=[json.loads(x) for x in (p/'rounds.jsonl').read_text().splitlines()]
        if done['rounds_completed']!=expected_rounds or [r['round'] for r in rows]!=list(range(expected_rounds)):
            raise ValueError(f'{mode} did not actually complete {expected_rounds} rounds')
        if expected_rounds==300 and not (done['formal_300_complete'] and m['formal_300']):
            raise ValueError('Smoke results cannot be treated as formal results')
        if m['mode']!=mode or done['mode']!=mode: raise ValueError('Mode mismatch')
        if ref is None: ref=m['protocol']
        if ref.get('evaluation',{}).get('profiles')!=['P6']:
            raise ValueError('Expected Full-only evaluation, not seven-profile test results')
        if m['protocol']!=ref or m['plan_sha256']!=content_hash(ref): raise ValueError('Paired plan mismatch')
        if done['final_source_sha256']!=ref['source_sha256']: raise ValueError('Training source changed')
        for r,entry in zip(rows,ref['schedule']):
            want=entry['random'] if mode=='C0' else entry['priority']
            if [(s['client'],s['profile']) for s in r['slots']]!=list(zip(entry['clients'],want)):
                raise ValueError('Actual dispatch differs from frozen plan')
            t=r.get('full_test',{})
            if t.get('profile')!='P6' or not all(isinstance(t.get(k),(int,float)) and math.isfinite(t[k]) for k in ('accuracy','ce')):
                raise ValueError('Missing/nonfinite Full-only test')
            if 'official_test' in r: raise ValueError('Unexpected old submodel evaluation schema')
            if mode in ('C0','C1') and r['reset_total']: raise ValueError('Unexpected reset')
            if r['round']<4 and r['reset_total']: raise ValueError('Reset before history warmup')
            if mode in ('C3','C4','C5') and r['reset_total'] and r['max_reset_multiplicity']!=10:
                raise ValueError('Shared reset concentration mismatch')
        data[mode]={'manifest':m,'rows':rows,'completed':done}
    for r in range(expected_rounds):
        for slot in range(10):
            refdiag=data['C2']['rows'][r]['slots'][slot]['kernels']
            for mode in ('C3','C4','C5'):
                diag=data[mode]['rows'][r]['slots'][slot]['kernels']
                if {k:x['count'] for k,x in diag.items()}!={k:x['count'] for k,x in refdiag.items()}:
                    raise ValueError('Reset counts differ')
        if len({data[m]['rows'][r]['reset_unique_segments'] for m in ('C3','C4','C5')})!=1:
            raise ValueError('Shared-position concentration differs')
    return data


def summarize(root):
    root=Path(root); data=load_suite(root); result={}
    for mode,run in data.items():
        rows=run['rows']; plan=run['manifest']['protocol']
        a=[r['full_test']['accuracy'] for r in rows]
        full={'test_oracle_peak':max(a),'peak_round':a.index(max(a))+1,'round300':a[-1],
              'last20_mean':statistics.mean(a[-20:]),'last50_mean':statistics.mean(a[-50:])}
        key='random_cost' if mode=='C0' else 'priority_cost'
        layers=[l for r in rows for s in r['slots'] for l in s['kernels'].values() if l['count']]
        n=sum(l['count'] for l in layers)
        result[mode]={'full':full,'Full_evaluation_count':len(rows),
            'full_slots':sum(r['profile_slots'][6] for r in rows),
            'profile_slots':[sum(r['profile_slots'][p] for r in rows) for p in range(7)],
            'unique_full_clients':len({s['client'] for r in rows for s in r['slots'] if s['profile']==6}),
            'train_MAC_proxy':sum(r['allocation'][key][0] for r in rows),
            'download_upload_bytes':sum(r['allocation'][key][1] for r in rows),
            'resets':sum(r['reset_total'] for r in rows),
            'first_reset_round':next((r['round']+1 for r in rows if r['reset_total']),None),
            'fallback_reset_count':sum(l['fallback_count'] for l in layers),
            'selected_instability_mean':sum(l['selected_instability_mean']*l['count'] for l in layers)/n if n else None,
            'unique_reset_segments_per_active_round':statistics.mean(r['reset_unique_segments'] for r in rows if r['reset_total']) if n else 0,
            'learning_delta_l2_mean':statistics.mean(s['learning_delta_l2'] for r in rows for s in r['slots']),
            'region_update_l2_mean':{k:statistics.mean(r['coverage_update_diagnostics'][k]['update_l2'] for r in rows[4:]) for k in rows[0]['coverage_update_diagnostics']},
            'timings_seconds':{k:sum(r['timings'][k] for r in rows) for k in rows[0]['timings']},
            'elapsed_seconds':run['completed']['elapsed_seconds']}
    comparisons={}
    for a,b in [('C1','C0'),('C2','C1'),('C3','C2'),('C4','C3'),('C5','C3'),('C4','C5')]:
        comparisons[f'{a}-{b}']={key:result[a]['full'][key]-result[b]['full'][key]
                                   for key in ('test_oracle_peak','round300','last20_mean','last50_mean')}
    obj={'completed_300_all_six':True,'arms':result,'Full_comparisons_pp':comparisons,
         'primary_metric':'Full test-oracle peak (official test exposed every round)',
         'limits':['Single seed1; no independent validation selection.','Do not tune using these official test results.',
                   'Costs approximately matched for C0/C1; MAC/communication are tensor/proxy estimates.',
                   'Concurrent CUDA timings and stochastic nondeterminism limit causal claims.']}
    (root/'summary.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (root/'full_accuracy_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['mode','test-oracle peak','peak_round','round300','last20_mean','last50_mean'])
        for mode,x in result.items():
            w.writerow([mode,*x['full'].values()])
    with (root/'full_accuracy_each_round.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['mode','round','Full_accuracy','Full_CE'])
        for mode,run in data.items():
            for row in run['rows']: w.writerow([mode,row['round']+1,row['full_test']['accuracy'],row['full_test']['ce']])
    lines=['# HGKR-FL 七档六组300轮汇总','',
           '七档P0–P6参与训练，每轮仅测试P6一次。六组均已核验真实完成300轮及300次Full测试。主观察指标为 Full **test-oracle peak**；它不是独立验证集选出的无偏测试性能。',
           '', '|模式|Full test-oracle peak|峰值轮|第300轮|末20轮|末50轮|Full槽位|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for mode,x in result.items():
        a=x['full']; lines.append(f"|{mode}|{a['test_oracle_peak']:.4f}|{a['peak_round']}|{a['round300']:.4f}|{a['last20_mean']:.4f}|{a['last50_mean']:.4f}|{x['full_slots']}|")
    lines+=['','## 预算和机制','', '|模式|七档槽位|MAC代理|通信GiB|重置次数|首次重置轮|历史不足回退|不同重置片段/活跃轮|训练秒|选择重置秒|服务器秒|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode,x in result.items():
        t=x['timings_seconds']
        lines.append(f"|{mode}|{x['profile_slots']}|{x['train_MAC_proxy']}|{x['download_upload_bytes']/2**30:.3f}|{x['resets']}|{x['first_reset_round']}|{x['fallback_reset_count']}|{x['unique_reset_segments_per_active_round']:.2f}|{t['local_train_seconds']:.1f}|{t['selection_reset_seconds']:.1f}|{t['server_seconds']:.1f}|")
    lines+=['','## 六组比较','']
    for pair,d in comparisons.items():
        lines.append(f"- {pair}：Full test-oracle peak差 {d['test_oracle_peak']:+.4f} 个百分点，第300轮 {d['round300']:+.4f}，末20轮 {d['last20_mean']:+.4f}，末50轮 {d['last50_mean']:+.4f}。")
    if comparisons['C4-C3']['test_oracle_peak']<=0:
        lines.append('\nC4历史高不稳定性选择未超过同轮共享随机C3的Full test-oracle peak；本轮没有证据按主要指标宣称该排序有收益。没有追加模块、调参或重试。')
    else:
        lines.append('\nC4按主要观察指标高于C3，是本次固定单seed300轮的探索性结果；同时检查Full末段、预算及C5，不将峰值提升表述为独立检验的无偏收益。')
    lines+=['','C1−C0检查预算内分配，C2−C1检查独立随机重置，C3−C2检查集中度，C4/C5−C3分别检查高/低不稳定性排序。历史方向波动不等于已证实语义冲突。',
            '', '每轮Full准确率见full_accuracy_each_round.csv，Full汇总见full_accuracy_summary.csv；成本、覆盖区域更新范数与时间见summary.json。不测试或报告P0–P5准确率。没有自动修改参数、重试seed或扩展训练轮数。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'SUCCESS.json').write_text(json.dumps({'all_six_completed_300':True}),encoding='utf-8')
    return obj


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('output_root'); summarize(p.parse_args().output_root)
