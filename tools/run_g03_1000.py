"""One fixed G03/seed1/1000-round run; reuse the original AdaptiveFL loop.

Separate entry only: the concurrently running P01-P10 source files stay intact.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
from tools import run_evidence_candidate as base


def fixed_args(output_root):
    original=sys.argv
    try:
        sys.argv=['main_fed_ori.py','--method','g03','--epochs','1000','--seed','1',
            '--gpu','0','--generate_data','0','--data_beta','.3','--eval-profiles','full',
            '--deterministic-execution','--no-save-pt','--method-run-name','G03_full_only_1000_seed1_deterministic',
            '--method-output-root',str(output_root)]
        args=base.args_parser()
    finally:sys.argv=original
    args.model='resnet';args.dataset='cifar100';args.algorithm='AdaptiveFL'
    return args


def read_rows(path,expected):
    with path.open() as f:rows=list(csv.DictReader(f))
    if [int(r['round']) for r in rows]!=list(range(1,expected+1)):
        raise ValueError('Incomplete round budget: '+str(path))
    if not all(math.isfinite(float(r[k])) for r in rows for k in ('full_accuracy','full_ce')):
        raise ValueError('Nonfinite evaluation')
    return rows


def summarize(name,rows):
    acc=[float(r['full_accuracy']) for r in rows];ce=[float(r['full_ce']) for r in rows]
    peak=acc.index(max(acc))
    return dict(id=name,rounds=len(rows),test_oracle_peak=acc[peak],best_round=int(rows[peak]['round']),
        final_accuracy=acc[-1],last20_mean=statistics.mean(acc[-20:]),last20_std=statistics.pstdev(acc[-20:]),
        last50_mean=statistics.mean(acc[-50:]),final_ce=ce[-1],peak_ce=ce[peak])


def report(path):
    rows=read_rows(path/'accuracy_each_round.csv',1000)
    done=json.loads((path/'completed.json').read_text());check=json.loads((path/'final_model_check.json').read_text())
    if not done['success'] or done['rounds_completed']!=1000 or done['checkpoint_policy']!='disabled_by_user':
        raise ValueError('Not actually completed 1000 rounds')
    if check['round']!=1000 or not check['finite_full_parameters'] or check['full_numel']!=11424356 or list(path.glob('*.pt')):
        raise ValueError('Final Full/no-checkpoint validation failed')
    with (path/'method_diagnostics.csv').open() as f:diags=list(csv.DictReader(f))
    if [int(r['round']) for r in diags]!=list(range(1,1001)):raise ValueError('Incomplete aggregation diagnostics')
    previous=ROOT/'runs/native_groups10_full400_seed1/g03/G03_full_only_400_seed1_deterministic/accuracy_each_round.csv'
    historical=read_rows(previous,400)
    records=[summarize('G03 historical 400',historical),summarize('G03 new first 400',rows[:400]),summarize('G03 new full 1000',rows)]
    output=path/'report';output.mkdir(exist_ok=False)
    with (output/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (output/'summary.json').write_text(json.dumps(dict(comparison=records,peak_label='official test-oracle peak',
        interpretation='Same G03 algorithm; 1000 versus 400 has more training and test-oracle opportunities, not an equal-budget algorithm gain',
        final_in_memory_check=check,save_pt=False),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key,label in [('full_accuracy','Full test Accuracy (%)'),('full_ce','Full test CE')]:
        fig,ax=plt.subplots(figsize=(11,5))
        ax.plot(range(1,1001),[float(r[key]) for r in rows],label='G03 1000 rounds',linewidth=1)
        ax.plot(range(1,401),[float(r[key]) for r in historical],label='Historical G03 400',linestyle='--',linewidth=1)
        ax.axvline(400,color='grey',linestyle=':',linewidth=.8);ax.set(xlabel='Round',ylabel=label)
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/(key+'.png'),dpi=150);plt.close(fig)
    lines=['# G03 fixed 1000-round run','',
        'All peaks are official test-oracle peaks. Different budgets are not an equal-budget algorithm comparison. No model weights saved.','',
        '| Run | Peak | Round | Final | Last20 | Last50 | Final CE |','|---|---:|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f"|{r['id']}|{r['test_oracle_peak']:.4f}|{r['best_round']}|{r['final_accuracy']:.4f}|{r['last20_mean']:.4f}|{r['last50_mean']:.4f}|{r['final_ce']:.6f}|")
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return records


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,default=Path('runs/g03_full1000_seed1_gpu0_nopt_20261021'))
    opt=parser.parse_args();os.chdir(ROOT)
    if os.environ['CUDA_VISIBLE_DEVICES']!='0':raise ValueError('This requested run uses physical GPU0 only')
    opt.output_root.mkdir(parents=True,exist_ok=False)
    args=fixed_args(opt.output_root)
    from utils.reproducibility import configure_deterministic_execution
    policy=configure_deterministic_execution(True);base.torch.set_num_threads(2);args.device=base.torch.device('cuda:0')
    path=base.prepare_output(args);started=time.perf_counter();stage='training'
    status=dict(pid=os.getpid(),state='running',method='g03',epochs=1000,seed=1,physical_gpu=0,save_pt=False,
        output=str(path),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    def save():
        status['updated']=time.time();pending=opt.output_root/'status.pending'
        pending.write_text(json.dumps(status,indent=2));pending.replace(opt.output_root/'status.json')
    save()
    try:
        base.set_random_seed(1)
        record=Path(base.data._build_record_path(args))
        args._method_partition_sha256=hashlib.sha256(record.read_bytes()).hexdigest()
        if args._method_partition_sha256!='41ab2d3486663ce0100e183ab00565de137f91af07632c236c2ddf60d890ba41':
            raise ValueError('Original partition differs')
        def loaded():
            transform=base.transforms.Compose([base.transforms.ToTensor(),base.transforms.Normalize((.5071,.4867,.4408),(.2675,.2565,.2761))])
            return (base.datasets.CIFAR100(str(opt.dataset_root),train=True,download=False,transform=transform),
                base.datasets.CIFAR100(str(opt.dataset_root),train=False,download=False,transform=transform))
        old=base.data.prepare_cifar100
        try:
            base.data.prepare_cifar100=loaded;train,test,users=base.data.get_dataset(args)
        finally:base.data.prepare_cifar100=old
        if len(train)!=50000 or len(test)!=10000 or len(users)!=100:raise ValueError('Complete original dataset required')
        (path/'run_environment.json').write_text(json.dumps(dict(pid=os.getpid(),argv=sys.argv,dataset_root=str(opt.dataset_root),
            cuda_visible_devices='0',gpu=base.torch.cuda.get_device_name(),torch=base.torch.__version__,
            train_examples=len(train),test_examples=len(test),deterministic_execution=policy,
            actual_entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),save_pt=False),indent=2))
        unused=base.ResNet18_cifar(num_channels=args.num_channels,num_classes=args.num_classes)
        base.torch.cuda.reset_peak_memory_stats()
        base.AdaptiveFL(args,train,test,users)
        stage='report';status['state']='training_completed';save()
        report(path);status['state']='completed';save()
    except BaseException:
        import traceback
        status.update(state='report_failed' if stage=='report' else 'failed',error=traceback.format_exc());save()
        (path/'failed.json').write_text(json.dumps({'stage':stage,'traceback':status['error']},indent=2));raise
    finally:
        if base.torch.cuda.is_initialized():
            (path/'resources.json').write_text(json.dumps(dict(wall_seconds=time.perf_counter()-started,
                peak_allocated_bytes=base.torch.cuda.max_memory_allocated(),peak_reserved_bytes=base.torch.cuda.max_memory_reserved()),indent=2))


if __name__=='__main__':main()
