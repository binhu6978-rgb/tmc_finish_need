from pathlib import Path

from tools.run_native_groups10 import jobs


def test_only_ten_fixed_jobs_in_requested_pairs_and_independent_paths():
    batches=jobs(Path('python'),Path('/dataset'),Path('runs/new_native_test'))
    assert [[j['id'] for j in batch] for batch in batches]==[
        ['G01','G02'],['G03','G06'],['G04','G05'],['G07','G08'],['G09','G10']]
    flat=sum(batches,[])
    assert len({j['output'] for j in flat})==10
    for j in flat:
        cmd=j['argv']
        for key,value in [('--epochs','400'),('--seed','1'),('--generate_data','0'),('--eval-profiles','full'),('--gpu','0')]:
            assert cmd[cmd.index(key)+1]==value
        assert '--deterministic-execution' in cmd
        assert '--conflict-lambda' not in cmd and '--local-reference' not in cmd
