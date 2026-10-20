"""Completed local REF/L01-L11 only; no Linux-peak-based ranking or tuning."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.report_tonight10 import validate,diagnostic_summary
from tools.report_next10_m2_m3 import describe


def report(root,output=None):
    cfg=json.loads((ROOT/'configs/e03_local12_400.json').read_text())
    records=[];accuracy={};ce={};mechanisms={};manifests=[];schedules=[]
    for name in sum(cfg['batches'],[]):
        method='m2' if name=='REF' else name.lower();path=root/method/(name+'_full_only_400_seed1')
        validate(path)
        r,curve=describe(name,method,path/'accuracy_each_round.csv',2)
        records.append(r);accuracy[name]=curve
        with (path/'accuracy_each_round.csv').open() as f:ce[name]=[float(x['full_ce']) for x in csv.DictReader(f)]
        mechanisms[name],schedule=diagnostic_summary(path/'method_diagnostics.csv');schedules.append(schedule)
        with (path/'method_diagnostics.csv').open() as f:raw=[json.loads(x['metrics_json'])['server'] for x in csv.DictReader(f)]
        mechanisms[name].update(normalized_share_changed_slots=sum(s.get('normalized_share_changed_slots',0) for s in raw),
            rounds_with_nonuniform_weights=sum(len(set(s['conflict_weights']))>1 for s in raw),
            rounds_with_e03_aggregate_change=sum(s.get('difference_from_e03_l2',0)>0 for s in raw),
            later_actual_update_l2_mean=statistics.mean(s['actual_update_l2'] for s in raw[1:]))
        manifests.append(json.loads((path/'manifest.json').read_text()))
    if any(s!=schedules[0] for s in schedules) or any(m['initial_state_sha256']!=manifests[0]['initial_state_sha256'] for m in manifests):
        raise ValueError('Local runs do not share actual initialization and slot schedule')
    ref=records[0]
    for r in records:
        for field in ('test_oracle_peak','final_accuracy','last20_mean','last50_mean'):
            r[field+'_minus_local_REF_pp']=r[field]-ref[field]
    output=output if output is not None else root/'report';output.mkdir(exist_ok=False)
    def csv_write(path,rows,fields):
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    csv_write(output/'comparison.csv',records,list(records[0]))
    (output/'summary.json').write_text(json.dumps({'comparison':records,'mechanisms':mechanisms,
        'primary_control':'local REF, exact original M2 lambda4','peak_label':'official-test test-oracle peak',
        'pairing':'all12 actual initialization and client/profile schedules match',
        'L02_note':'Algebraically identical REF after synchronized dispatch; differences cannot be attributed to a new post-sync mechanism'},indent=2))
    # fd Library/bin is in PATH by the local queue, as in an activated conda env.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for data,label,filename in ((accuracy,'Full test Accuracy (%)','accuracy'),(ce,'Full test CE','ce')):
        csv_write(output/(filename+'_all_rounds.csv'),[{'round':i+1,**{n:v[i] for n,v in data.items()}} for i in range(400)],['round',*data])
        fig,ax=plt.subplots(figsize=(12,6))
        for name,values in data.items():ax.plot(range(1,401),values,label=name,linewidth=2 if name=='REF' else 1,linestyle='--' if name=='REF' else '-')
        ax.set(xlabel='Communication round',ylabel=label);ax.grid(alpha=.2);ax.legend(ncol=6);fig.tight_layout();fig.savefig(output/(filename+'.png'),dpi=170);plt.close(fig)
    # L02 is algebraically REF: retain it in the table as a numeric control,
    # rather than ranking it as an independently changed algorithm.
    candidates=[r for r in records[1:] if r['id']!='L02']
    peak=max(candidates,key=lambda r:r['test_oracle_peak']);late=max(candidates,key=lambda r:r['last20_mean'])
    text=['# 本地E03参考与11组固定机制实验','',
        '全部12组完成400轮、seed1。主要对照为本地REF，不以Linux E03的37.29%作为优劣判据。峰值均为官方测试集test-oracle peak；单种子探索不等于独立验证的泛化优势。','',
        '|方法|test-oracle peak %|轮次|Final %|Last20 %|Last50 %|Final CE|Peak−REF pp|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:text.append(f"|{r['id']}|{r['test_oracle_peak']:.4f}|{r['best_round']}|{r['final_accuracy']:.4f}|{r['last20_mean']:.4f}|{r['last50_mean']:.4f}|{r['final_ce']:.6f}|{r['test_oracle_peak_minus_local_REF_pp']:+.4f}|")
    text+=['',f"排除代数等价中性对照L02后，新机制最高观察峰值：{peak['id']}，{peak['test_oracle_peak']:.4f}%；是否超过本地REF：{'是' if peak['test_oracle_peak']>ref['test_oracle_peak'] else '否'}。",
        f"非中性新机制最高Last20：{late['id']}，{late['last20_mean']:.4f}%；REF Last20为{ref['last20_mean']:.4f}%。",
        '', 'L02在原全七档同步后与REF代数等价，是数值一致性对照，不构成新机制；CUDA独立进程可能存在原有非确定性。',
        'L01隔离的主要是首轮独立初始化融合。L03/L04/L05保护的参数类由真实模块识别；L06保护物理能力ID，L07保护实际Full槽位，两者不是同一划分。',
        'L09先等权平均有效分块余弦再取负风险。L10对其他物理客户端内的有效slot余弦取中位数，再对不同客户端中位数取中位数；不同物理客户端不重复计数。',
        '机制日志的E03更新比较使用各方法自身上传，不是额外重跑REF轨迹。首轮融合与第2–400轮L2分开统计，不能仅凭峰值或Gate触发认定有效。','',
        '![Full Accuracy](accuracy.png)','', '![Full CE](ce.png)']
    (output/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();report(a.root,a.output)
