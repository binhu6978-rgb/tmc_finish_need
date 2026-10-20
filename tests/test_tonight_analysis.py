import pytest
from tools.analyze_tonight_results import decomposition


def test_exact_scalar_rescaling_has_no_perpendicular_change():
    result=decomposition([dict(effective_step_along_e03=.98,e03_update_l2=2.,actual_update_l2=1.96)])
    assert result['parallel_difference_energy']==pytest.approx(.0016)
    assert result['perpendicular_fraction']==pytest.approx(0.,abs=1e-10)


def test_perpendicular_component_is_not_misidentified_as_scalar_step():
    result=decomposition([dict(effective_step_along_e03=1.,e03_update_l2=2.,actual_update_l2=5**.5)])
    assert result['parallel_difference_energy']==0.
    assert result['perpendicular_difference_energy']==pytest.approx(1.)
    assert result['perpendicular_fraction']==pytest.approx(1.)
