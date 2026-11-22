"""Tests of the model-versus-referee diagnostic.

The point of the diagnostic is to catch a model that does not rank like the referee, so the
tests that matter are the ones where it must say NO: a model anti-correlated with the referee,
a weak model, and a model whose optimum is a plateau of solutions the referee scores very
differently. The polarity tests exist because getting the sign backwards is the mistake this
module was written to prevent.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics import MIN_TIES, STRONG_RHO, baseline_table, rank_agreement  # noqa: E402

BOTH_MIN = dict(model_lower_is_better=True, referee_lower_is_better=True)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ------------------------------------------------------------------------------- polarity


def test_both_minimised_agreement_is_positive(rng):
    referee = rng.uniform(0, 100, 200)
    model = referee + rng.normal(0, 3, 200)          # a good model, same direction
    out = rank_agreement(model, referee, **BOTH_MIN)
    assert out.rho > 0.9 and out.significant and out.strong


def test_maximised_model_is_oriented_before_correlating(rng):
    """A score where HIGHER is better must come out as agreement too, not as disagreement."""
    referee = rng.uniform(0, 100, 200)
    model = -(referee + rng.normal(0, 3, 200))       # same information, opposite convention
    out = rank_agreement(model, referee, model_lower_is_better=False, referee_lower_is_better=True)
    assert out.rho > 0.9 and out.significant and out.strong


def test_the_two_polarity_flags_are_not_interchangeable(rng):
    referee = rng.uniform(0, 100, 200)
    model = referee + rng.normal(0, 3, 200)
    right = rank_agreement(model, referee, **BOTH_MIN)
    wrong = rank_agreement(model, referee, model_lower_is_better=True, referee_lower_is_better=False)
    assert right.rho == pytest.approx(-wrong.rho)
    assert right.significant and not wrong.significant


def test_polarity_flags_are_mandatory(rng):
    with pytest.raises(TypeError):
        rank_agreement(rng.normal(size=10), rng.normal(size=10))


def test_a_maximised_referee_reports_its_best_in_its_own_units():
    """With referee_lower_is_better=False the best tie is the LARGEST score, not the smallest."""
    model = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    referee = np.array([90.0, 70.0, 10.0, 20.0, 30.0])      # higher is better
    out = rank_agreement(model, referee, model_lower_is_better=True, referee_lower_is_better=False)
    assert out.referee_at_model_optimum == pytest.approx(90.0)
    assert out.referee_mean_at_model_optimum == pytest.approx(80.0)
    assert out.referee_best == pytest.approx(90.0)


# -------------------------------------------------------------------- it must be able to say no


def test_a_useless_model_does_not_agree(rng):
    referee = rng.uniform(0, 100, 300)
    model = rng.uniform(0, 100, 300)                 # unrelated
    out = rank_agreement(model, referee, **BOTH_MIN)
    assert not out.significant and not out.strong and out.verdict == "weak"
    assert abs(out.rho) < 0.2


def test_an_anticorrelated_model_does_not_agree(rng):
    referee = rng.uniform(0, 100, 200)
    model = -referee + rng.normal(0, 1, 200)         # systematically prefers the worst
    out = rank_agreement(model, referee, **BOTH_MIN)
    assert out.rho < -0.9 and not out.significant and not out.strong


def test_a_weak_but_significant_agreement_is_not_reported_as_agreement(rng):
    """Significance is not the operational threshold.

    With 216 solutions this model's rho (about 0.4) clears p < 0.05 easily and is still far
    below STRONG_RHO, so `strong` must stay False and the printed verdict must say so.
    """
    n = 216
    referee = rng.uniform(0, 100, n)
    model = 0.16 * referee + rng.normal(0, 10, n)      # weak but real
    out = rank_agreement(model, referee, **BOTH_MIN)
    assert 0.0 < out.rho < STRONG_RHO
    assert out.significant and not out.strong
    assert out.verdict == "weak"
    assert "TOO WEAK" in str(out)


def test_a_strong_agreement_is_reported_as_such(rng):
    referee = rng.uniform(0, 100, 216)
    model = referee + rng.normal(0, 5, 216)
    out = rank_agreement(model, referee, **BOTH_MIN)
    assert out.strong and out.verdict == "strong"


# ----------------------------------------------------------------------------- degeneracy


def test_a_plateau_under_a_strong_ranking_is_underdetermined():
    """Twenty solutions tied at the model's minimum, scored differently by the referee."""
    model = np.concatenate([np.zeros(20), np.arange(1.0, 81.0)])
    referee = np.concatenate([np.linspace(10.0, 30.0, 20), np.linspace(31.0, 130.0, 80)])
    out = rank_agreement(model, referee, **BOTH_MIN)
    assert out.strong
    assert out.n_tied_at_model_optimum == 20
    assert out.referee_spread_at_model_optimum == pytest.approx(20.0)
    assert out.referee_at_model_optimum == pytest.approx(10.0)
    assert out.referee_mean_at_model_optimum == pytest.approx(20.0)
    assert out.verdict == "underdetermined"
    assert "UNDERDETERMINED" in str(out)


def test_a_weak_model_with_a_plateau_is_reported_as_weak_first(rng):
    """Breaking ties cannot repair a bad ranking, so 'weak' must win over 'underdetermined'."""
    model = np.concatenate([np.zeros(20), rng.uniform(1.0, 100.0, 80)])
    referee = np.concatenate([np.linspace(10.0, 30.0, 20), rng.uniform(10.0, 130.0, 80)])
    out = rank_agreement(model, referee, **BOTH_MIN)
    assert out.rho < STRONG_RHO and out.degenerate
    assert out.verdict == "weak"
    assert "TOO WEAK" in str(out) and "fix the ranking first" in str(out)


def test_a_few_ties_are_not_a_plateau():
    referee = np.arange(100.0)
    model = referee.copy()
    model[:3] = 0.0                                    # 3 ties at the optimum, referee 0, 1, 2
    out = rank_agreement(model, referee, **BOTH_MIN)
    assert out.n_tied_at_model_optimum == 3 < MIN_TIES
    assert out.verdict == "strong"


def test_a_sharp_optimum_is_not_flagged_as_degenerate(rng):
    referee = rng.uniform(0, 100, 100)
    out = rank_agreement(referee.copy(), referee, **BOTH_MIN)
    assert out.n_tied_at_model_optimum == 1
    assert not out.degenerate and out.verdict == "strong"


def test_tolerance_groups_energies_that_only_differ_by_rounding():
    model = np.array([1.0, 1.0 + 1e-12, 1.0 + 2e-12, 5.0, 6.0])
    referee = np.array([3.0, 7.0, 9.0, 1.0, 2.0])
    tight = rank_agreement(model, referee, **BOTH_MIN)
    loose = rank_agreement(model, referee, **BOTH_MIN, tolerance=1e-9)
    assert tight.n_tied_at_model_optimum == 1
    assert loose.n_tied_at_model_optimum == 3


# ------------------------------------------- ties are reported honestly, not optimistically


def test_the_referee_mean_over_tied_optima_is_reported_beside_the_best():
    """A sampler picks among the model's tied optima, so the best of them flatters the model."""
    model = np.concatenate([np.zeros(4), np.ones(10)])
    referee = np.concatenate([np.array([10.0, 20.0, 30.0, 40.0]), np.full(10, 50.0)])
    out = rank_agreement(model, referee, **BOTH_MIN)
    assert out.referee_at_model_optimum == pytest.approx(10.0)
    assert out.referee_mean_at_model_optimum == pytest.approx(25.0)


def test_rank_of_best_finds_a_model_that_buries_the_true_optimum():
    referee = np.array([0.0, 1.0, 2.0, 3.0, 4.0])    # solution 0 is the referee's best
    model = np.array([9.0, 1.0, 2.0, 3.0, 4.0])      # the model ranks it last
    out = rank_agreement(model, referee, **BOTH_MIN)
    assert out.rank_of_best == 5


def test_rank_of_best_shows_the_width_of_a_tie():
    referee = np.array([0.0, 1.0, 2.0, 3.0, 4.0])      # solution 0 is the referee's best
    model = np.array([5.0, 5.0, 5.0, 1.0, 2.0])        # it ties with two others
    out = rank_agreement(model, referee, **BOTH_MIN)
    assert out.rank_of_best == 3 and out.n_tied_with_best == 3
    assert "3-5 (tied)" in str(out)


def test_rank_of_best_does_not_depend_on_the_order_of_shared_referee_optima():
    """Two solutions share the referee's best score; the model ranks one last, one first."""
    referee = np.array([0.0, 0.0, 5.0, 6.0, 7.0])
    model = np.array([9.0, 1.0, 2.0, 3.0, 4.0])
    swapped = [1, 0, 2, 3, 4]
    a = rank_agreement(model, referee, **BOTH_MIN)
    b = rank_agreement(model[swapped], referee[swapped], **BOTH_MIN)
    assert a.rank_of_best == b.rank_of_best == 1
    assert a.n_referee_optima == 2


# --------------------------------------------------------------------------------- misuse


def test_mismatched_lengths_are_refused(rng):
    with pytest.raises(ValueError, match="same length"):
        rank_agreement(rng.normal(size=10), rng.normal(size=11), **BOTH_MIN)


def test_a_nan_is_refused_instead_of_ranked():
    model = np.array([1.0, np.nan, 3.0, 4.0])
    with pytest.raises(ValueError, match="NaN"):
        rank_agreement(model, np.arange(4.0), **BOTH_MIN)


def test_too_few_solutions_to_rank(rng):
    with pytest.raises(ValueError, match="at least 3"):
        rank_agreement([1.0, 2.0], [3.0, 4.0], **BOTH_MIN)


# ------------------------------------------------------------------------- baseline table


def test_baseline_table_puts_the_best_first_and_shows_the_gap():
    text = baseline_table(**{"random": 300000.0, "simulated annealing": 280000.0, "QAOA": 290000.0})
    lines = text.splitlines()
    assert lines[0].startswith("simulated annealing")
    assert "%" not in lines[0]          # the best row carries no gap
    assert all("%" in line for line in lines[1:])


def test_baseline_table_handles_a_zero_best_without_dividing_by_it():
    text = baseline_table(a=0.0, b=5.0)
    assert text.splitlines()[1].rstrip().endswith("+5")


def test_baseline_table_refuses_to_report_nothing():
    with pytest.raises(ValueError):
        baseline_table()
