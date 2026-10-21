"""Actual seven ResNet shapes and class heads; synthetic uploads, no FL run."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    opt = parser.parse_args()
    import torch
    from tools.run_final40_job import make_args
    from capability_final40 import Final40Hooks, active_cells
    from models.Fed import get_model_list, Aggregation_AdaptiveFL, split_model
    from utils.reproducibility import configure_deterministic_execution
    from utils.set_seed import set_random_seed
    from semcons_scope import slices
    configure_deterministic_execution(True)
    torch.set_num_threads(2)
    records = {}
    torch.cuda.reset_peak_memory_stats()
    for condition in ('A', 'S1', 'S2', 'S3', 'S4', 'S5'):
        names = [f'A{i:02}' for i in range(1, 21)] if condition == 'A' else [condition + '-' + method for method in ('Native', 'G03', 'P04', 'P05')]
        args, _, _ = make_args(names[0], Path('runs/not_created'), opt.prepared)
        args.device = torch.device('cuda:0')
        set_random_seed(1)
        with contextlib.redirect_stdout(io.StringIO()):
            models, _ = get_model_list(args)
        # Every class count actually forwards/backprops all original profiles.
        if condition in ('A', 'S5'):
            images = torch.linspace(-1, 1, 4 * 3 * 32 * 32, device='cuda').reshape(4, 3, 32, 32)
            for model in models:
                out = model(images)['output']
                assert out.shape == (4, args.num_classes)
                out.square().mean().backward()
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                model.zero_grad(set_to_none=True)
        previous = copy.deepcopy(models[-1].state_dict())
        profiles = [0, 1, 2, 3, 4, 5, 6, 6, 6, 6]
        lo, hi = args._final40_context['boundaries']
        cids = [1, 2, 3, lo, lo+1, lo+2, hi, hi, hi+1, hi+2]
        lens = [100 + cid for cid in cids]
        uploads = [{k: v.clone() + (i-3) * .001 for k, v in split_model(previous, models[p].state_dict()).items()} for i, p in enumerate(profiles)]
        native = Aggregation_AdaptiveFL(uploads, lens, previous)
        for name in names:
            args, _, _ = make_args(name, Path('runs/not_created'), opt.prepared)
            hook = Final40Hooks(args, models)
            hook.set_slots(zip(cids, profiles))
            first = hook.aggregate(uploads, lens, previous)
            assert all(torch.equal(first[k], native[k]) for k in native)
            del first
            out = hook.aggregate(uploads, lens, previous)
            diag = hook.round_diagnostics
            assert len(hook.regions) == 9 and all(bool(torch.isfinite(v).all()) for v in out.values())
            for region in hook.regions:
                if name in [f'A{i:02}' for i in range(5, 11)]:
                    selected = active_cells(args.method, region, hook.bn_keys)
                    for cell in region['cells']:
                        if cell not in selected:
                            assert torch.equal(out[cell['key']][slices(cell)], native[cell['key']][slices(cell)])
            if name in ('A01', 'A02', 'A03', 'A04'):
                assert diag['matched_norm_relative_error'] < 1e-5
            records[name] = dict(classes=args.num_classes, regions=len(hook.regions),
                difference_from_native_l2=diag['difference_from_native_l2'],
                norm_matching_relative_error=diag['matched_norm_relative_error'],
                population=hook.population, boundaries=hook.boundaries,
                full_numel=sum(v.numel() for v in out.values()))
            del out, hook
        del uploads, previous, native, models
    opt.output.write_text(json.dumps(dict(type='actual shapes and synthetic uploads, not a CIFAR performance experiment',
        checks=records, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20), indent=2))


if __name__ == '__main__':
    main()
