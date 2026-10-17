"""Offline scalar summaries only; never chooses hyperparameters or launches training."""
import collections
import json
from pathlib import Path
import statistics
import sys


def mean(values): return statistics.mean(list(values))


def summarize(root):
    root=Path(root)
    arms={}
    all_rows={}
    for mode in ('B0','B1','B2','B3'):
        output=root/f'pair100_{mode}'
        manifest=json.loads((output/'manifest.json').read_text())
        rows=[json.loads(x) for x in (output/'rounds.jsonl').read_text().splitlines()]
        all_rows[mode]=rows
        if [r['round'] for r in rows]!=list(range(100)): raise ValueError('Only summarize completed 100 rounds')
        accuracy={}
        for p,name in enumerate(('Small','Medium','Full')):
            values=[r['official_test'][p]['accuracy'] for r in rows]
            accuracy[name]={'last':values[-1],'last20_mean':mean(values[-20:]),
                            'test_oracle_peak':max(values),'peak_round_1based':values.index(max(values))+1}
        timings={k:sum(r['timings'][k] for r in rows) for k in rows[0]['timings']}
        slot_totals=[sum(r['profile_slots'][p] for r in rows) for p in range(3)]
        layers=[layer for r in rows for s in r['slots'] for layer in s['kernels'].values() if layer['count']]
        full_clients=collections.Counter(s['client'] for r in rows for s in r['slots'] if s['profile']==2)
        cost_key='B0_cost' if mode=='B0' else 'priority_cost'
        cost=[sum(r['allocation'][cost_key][i] for r in rows) for i in (0,1)]
        arms[mode]={'accuracy':accuracy,'timings_seconds':timings,'profile_slots':slot_totals,
                    'train_forward_MAC_proxy':cost[0],'train_FLOP_proxy':6*cost[0],
                    'download_upload_bytes':cost[1],'full_client_exposure':dict(full_clients),
                    'unique_full_clients':len(full_clients),'resets':sum(r['reset_total'] for r in rows),
                    'first_reset_round_1based':next((r['round']+1 for r in rows if r['reset_total']),None),
                    'history_fallback_selections':sum(l['fallback_count'] for l in layers),
                    'selected_instability_mean':mean(l['selected_instability_mean'] for l in layers) if layers else None,
                    'reset_L2_mean':mean(s['reset_l2'] for r in rows for s in r['slots']),
                    'learning_delta_L2_mean':mean(s['local_learning_delta_l2'] for r in rows for s in r['slots']),
                    'region_update_L2_mean':{k:mean(r['coverage_update_diagnostics'][k]['update_l2'] for r in rows) for k in rows[0]['coverage_update_diagnostics']},
                    'git_head':manifest['git_head'],'output':str(output.resolve())}
    overlap=total=0
    for a,b in zip(all_rows['B2'],all_rows['B3']):
        for sa,sb in zip(a['slots'],b['slots']):
            for key,layer in sa['kernels'].items():
                aa=set(layer['indices']); bb=set(sb['kernels'][key]['indices'])
                overlap+=len(aa&bb); total+=len(aa)
    result={'arms':arms,'B2_B3_selection_overlap':overlap,'B2_selection_count':total,
            'B2_B3_selection_overlap_fraction':overlap/total if total else None,
            'paired_Full_last20_differences_pp':{f'{a}-{b}':arms[a]['accuracy']['Full']['last20_mean']-arms[b]['accuracy']['Full']['last20_mean'] for a,b in [('B3','B2'),('B2','B1'),('B1','B0')]},
            'limits':['Single seed, 100 rounds, exploratory.','Every peak is test-oracle peak.',
                      'Budget matching is approximate (2% per-round tolerance), FLOPs are a conv/linear proxy.',
                      'Concurrent GPU timings are contention-affected, not isolated benchmarks.',
                      'Different trajectories and CUDA nondeterminism prevent a definitive semantic/causal attribution.',
                      'No 1000-round run is authorized or launched.']}
    (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# HGKR-FL 四组100轮真实实验汇总','',
           'seed=1，三档协议；每轮评估原始官方测试集。所有峰值均为 **test-oracle peak**。',
           '', '|模式|Full末轮|Full末20轮均值|test-oracle peak|峰值轮次|Small末20轮|Medium末20轮|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for mode,a in arms.items():
        f=a['accuracy']['Full']
        lines.append(f"|{mode}|{f['last']:.4f}|{f['last20_mean']:.4f}|{f['test_oracle_peak']:.4f}|{f['peak_round_1based']}|{a['accuracy']['Small']['last20_mean']:.4f}|{a['accuracy']['Medium']['last20_mean']:.4f}|")
    lines+=['','## 机制与成本','', '|模式|S/M/F槽位|重置片段数|首次重置轮|历史不足回退|训练秒|选择重置秒|服务器秒|通信GiB|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode,a in arms.items():
        t=a['timings_seconds']
        lines.append(f"|{mode}|{a['profile_slots']}|{a['resets']}|{a['first_reset_round_1based']}|{a['history_fallback_selections']}|{t['local_train_seconds']:.1f}|{t['selection_reset_seconds']:.1f}|{t['server_seconds']:.1f}|{a['download_upload_bytes']/2**30:.3f}|")
    lines+=['',f"B2/B3 选中片段的交集占 B2 数量：{result['B2_B3_selection_overlap_fraction']}。这是实际不同轨迹的选择差异，不是固定上传下的效用检验。",'',
            '## 配对差值及解释边界','']
    for key,value in result['paired_Full_last20_differences_pp'].items(): lines.append(f'- {key} Full末20轮均值差：{value:+.4f} 个百分点。')
    lines+=['','B3−B2 是历史选择的主要探索性比较；B2−B1 检查随机重置；B1−B0 检查分配策略。必须同时阅读三档准确率、成本和覆盖区域更新范数。Gate 活跃、重置活跃或单次峰值不构成算法有效性证明。',
            '', '历史方向波动仅是参数优化统计，尚未证实语义冲突或更新有害性。单种子及CUDA非确定性不支持稳健因果结论；没有根据测试集自动调参。',
            '', '完整三档统计、Full客户端暴露分布、计算量代理值、覆盖区域更新范数及真实开销见 summary.json。',
            '', '不自动启动1000轮。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__': summarize(sys.argv[1])
