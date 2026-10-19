"""Offline reporting only; never trains, selects parameters or changes results."""
import csv
import json
import math
from pathlib import Path
import shlex
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read_rounds(path):
    with Path(path).open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 400 or [int(r['round']) for r in rows] != list(range(1,401)):
        raise ValueError(f'Expected completed 400-round data: {path}')
    values = [float(r['full_accuracy']) for r in rows]
    ce = [float(r['full_ce']) for r in rows]
    if not all(math.isfinite(v) for v in values+ce):
        raise ValueError(f'Nonfinite report data: {path}')
    return rows, values, ce


def describe(name, method, path, radius):
    rows, values, ce = read_rounds(path)
    best = values.index(max(values))
    start, end = max(0,best-radius), min(len(values),best+radius+1)
    return {'id':name,'method':method,'rounds':len(rows),
            'test_oracle_peak':values[best],'best_round':best+1,
            'final_accuracy':values[-1],'last20_mean':statistics.mean(values[-20:]),
            'last50_mean':statistics.mean(values[-50:]),'final_ce':ce[-1],
            'peak_ce':ce[best],'peak_ce_window_start':start+1,'peak_ce_window_end':end,
            'peak_window_ce_mean':statistics.mean(ce[start:end])}, values


def report_next10(batches, output, config):
    output = Path(output)
    radius = config['peak_ce_window_radius']
    references = []
    reference_curves = {}
    for ref in config['references']:
        item, curve = describe(ref['id'],'historical',ROOT/ref['rounds_csv'],radius)
        references.append(item);reference_curves[ref['id']] = curve
    records = [];curves = {}
    for batch in batches:
        for job in batch:
            item, curve = describe(job['id'],job['method'],Path(job['output_dir'])/'accuracy_each_round.csv',radius)
            item['command'] = shlex.join(job['argv'])
            for ref in references:
                item['peak_minus_'+ref['id']+'_pp'] = item['test_oracle_peak']-ref['test_oracle_peak']
            records.append(item);curves[job['id']] = curve
    plot_dir = output/'accuracy_curves'
    plot_dir.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rounds = list(range(1,401))
    fig, ax = plt.subplots(figsize=(12,6))
    for name, values in curves.items():
        ax.plot(rounds,values,label=name,linewidth=1)
    for name, values in reference_curves.items():
        ax.plot(rounds,values,label=name+' historical',linestyle='--',linewidth=1.6)
    ax.set(xlabel='Communication round',ylabel='Full test Accuracy (%)',title=config.get('title','M2/M3 exploratory sweep: Full Model'))
    ax.legend(ncol=4,fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(output/'full_accuracy_comparison.png',dpi=180);plt.close(fig)
    for name, values in curves.items():
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(rounds,values,linewidth=1)
        for ref in references:
            ax.axhline(ref['test_oracle_peak'],label=ref['id']+' historical test-oracle peak',linestyle='--',linewidth=1)
        ax.set(xlabel='Communication round',ylabel='Full test Accuracy (%)',title=name+' Full Accuracy')
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(plot_dir/(name+'.png'),dpi=150);plt.close(fig)
    with (output/'comparison.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    with (output/'full_accuracy_all_rounds.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['round',*curves]);writer.writeheader()
        writer.writerows({'round':r,**{name:v[r-1] for name,v in curves.items()}} for r in rounds)
    (output/'reference_metrics.json').write_text(json.dumps(references,indent=2)+'\n',encoding='utf-8')
    text=['# '+config.get('title','Next10 M2/M3: 400-round exploratory results'),'',
          'All peaks are official-test **test-oracle peaks**, not independent validation-selected performance.',
          'This second sweep was chosen after inspecting the previous official-test results; it is further exploration, not an independent confirmation.','',
          'Peak-neighborhood CE is the arithmetic mean in the fixed peak round +/- 2 window, clipped to rounds 1..400.',
          'Peak CE itself is also reported. Tied maxima use the first round.','',
          '|ID|Method|test-oracle peak|Round|Final|Last20|Last50|Final CE|Peak CE|Peak-window CE|'+''.join(ref['id']+' delta pp|' for ref in references),
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|'+('---:|'*len(references))]
    for r in records:
        text.append('|{id}|{method}|{test_oracle_peak:.4f}|{best_round}|{final_accuracy:.4f}|{last20_mean:.4f}|{last50_mean:.4f}|{final_ce:.6f}|{peak_ce:.6f}|{peak_window_ce_mean:.6f}|'.format(**r)+''.join(f"{r['peak_minus_'+ref['id']+'_pp']:+.4f}|" for ref in references))
    for ref in references:
        text += ['',f"Historical {ref['id']}: peak {ref['test_oracle_peak']:.4f}% at round {ref['best_round']}; final {ref['final_accuracy']:.4f}%. Read directly from preserved CSV."]
    best=max(records,key=lambda r:r['test_oracle_peak'])
    text += ['',f"Highest observed next10 test-oracle peak: {best['id']} = {best['test_oracle_peak']:.4f}% (round {best['best_round']}).",
             'A higher peak is an exploratory observation from one seed. It does not establish unbiased performance or a robust improvement.',
             '', '![All Full Accuracy curves](full_accuracy_comparison.png)',
             '', 'Individual curves: accuracy_curves/<ID>.png. Raw 400-round values: full_accuracy_all_rounds.csv.',
             'Per-round CE, cumulative peak, timings and M2/M3 diagnostics remain in each experiment directory. Best/latest checkpoints are preserved.']
    (output/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
