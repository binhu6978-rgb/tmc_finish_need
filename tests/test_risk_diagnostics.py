"""Pure scalar/history tests; no training, CUDA operations or datasets."""
from dataclasses import replace

import pytest

from research.risk_diagnostics import ShadowMemory, fixed_effect_residuals, rho
from semconsfl.config import SemanticConfig


def config():
    return replace(SemanticConfig(), min_profile_records=2, min_client_records=1)


def test_ecdf_uses_midrank_and_owns_past_client_history():
    memory = ShadowMemory(config(), "ecdf_rank")
    memory.profiles = {0: [(1, 1, 0.), (1, 2, 1.)]}
    # Formal-history z could be zero; the candidate must retain its own old q.
    memory.clients = {8: [(1, .75)]}
    decision = memory.decide(8, 0, .5, 2, memory.snapshot())
    assert decision["z"] == .5
    assert decision["r"] == .25
    assert decision["gamma"] == 1 - .25 * .5 * .25
    original = memory.snapshot()
    memory.update(2, [(8, 0, .5, decision)])
    assert original[1][8] == [(1, .75)]
    assert memory.clients[8][-1] == (2, .5)


def test_short_profile_horizon_does_not_shorten_physical_client_history():
    memory = ShadowMemory(config(), "short20")
    memory.profiles = {0: [(1, 1, 100.), (2, 2, 100.), (20, 3, 0.), (21, 4, .1)]}
    memory.clients = {8: [(3, .6)]}
    decision = memory.decide(8, 0, .2, 25, memory.snapshot())
    assert decision["profile_records"] == 2 and decision["profile_median"] == .05
    assert decision["client_records"] == 1
    assert decision["r"] == pytest.approx(.2)
    assert decision["z"] == 1


@pytest.mark.parametrize("method", ["formal", "ecdf_rank", "short20"])
def test_candidates_reject_current_or_future_profile_history(method):
    memory = ShadowMemory(config(), method)
    memory.profiles = {0: [(1, 1, .1), (5, 2, .2)]}
    with pytest.raises(ValueError, match="leaked"):
        memory.decide(8, 0, .3, 5, memory.snapshot())


def test_zero_iqr_equality_is_not_positive_formal_z_and_rank_is_not_harm_label():
    formal = ShadowMemory(config(), "formal")
    ranked = ShadowMemory(config(), "ecdf_rank")
    for memory in (formal, ranked):
        memory.profiles = {0: [(1, 1, 0.), (1, 2, 0.)]}
        memory.clients = {8: [(1, .5)]}
    assert formal.decide(8, 0, 0., 2, formal.snapshot())["z"] == 0
    # Diagnostic warning example: tied zero anomalies have percentile .5,
    # which is not evidence of harmfulness. ECDF is NOT a validated Gate.
    assert ranked.decide(8, 0, 0., 2, ranked.snapshot())["z"] == .5


def test_duplicate_observations_mean_within_profile_then_across_profiles():
    memory = ShadowMemory(config(), "ecdf_rank")
    memory.update(2, [(8, 0, .2, {"z": 0.}), (8, 0, .4, {"z": 1.}),
                      (8, 1, .6, {"z": 1.})])
    assert memory.profiles[0] == [(2, 8, pytest.approx(.3))]
    assert memory.clients[8] == [(2, .75)]


def test_rank_constant_variables_stay_undefined():
    assert rho([1, 1, 1], [1, 2, 3]) is None
    assert rho([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)


def test_fixed_effect_projection_matches_exact_balanced_two_way_solution():
    residual, _, error = fixed_effect_residuals([1, 2, 3, 8], [0, 0, 1, 1], [0, 1, 0, 1])
    assert residual.tolist() == pytest.approx([1, -1, -1, 1])
    assert error < 1e-10


def test_fixed_effect_projection_has_zero_group_means_on_unbalanced_data():
    residual, _, error = fixed_effect_residuals([1, 2, 3, 8, 10, 5], [0, 0, 1, 1, 1, 2], [0, 1, 0, 1, 0, 1])
    assert sum(residual) == pytest.approx(0, abs=1e-10)
    assert error < 1e-10
