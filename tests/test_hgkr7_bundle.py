"""Archive layout must follow the local project, including original data helpers."""
from tools.build_hgkr7_bundle import is_project_source, project_files


def test_full_project_source_layout():
    paths=set(project_files())
    assert {'utils/get_datatset_new.py','utils/tinyimagenet.py','main_fed_ori.py',
            'Algorithm/Training_AdaptiveFL.py','experiments/run_semconsfl.py',
            'experiments/run_hgkr7.py','hgkr/seven.py','run_7tier_300.sh',
            'configs/hgkr7/C0.json','data/cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json'} <= paths
    assert not any('__pycache__' in p or p.endswith('.pyc') for p in paths)


def test_results_private_artifacts_and_cache_are_not_sources():
    for name in ('runs/check/summary.json','tmp/private.txt','models/__pycache__/resnet.pyc',
                 '.vscode/settings.json','paper.pdf','review.docx','data/cifar100/train','model.pt'):
        assert not is_project_source(name)
    assert is_project_source('utils/get_datatset_new.py')
    assert is_project_source('models/mobileNetV2 copy.py')
