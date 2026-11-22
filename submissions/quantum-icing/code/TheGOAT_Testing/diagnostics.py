"""Does a surrogate rank solutions the way the referee does?

A QUBO (or any surrogate cost) is a *model* of the official objective, never the objective
itself. Improving the sampler only pays off when the model ranks solutions the way the referee
does; when it does not, a perfect optimum of the model is worth no more than a random feasible
point. This module measures that agreement on a set of solutions scored by both.

Polarity is a mandatory argument because getting it backwards is an easy and costly mistake: a
rank correlation flips sign with the convention. Both inputs are oriented so that SMALLER is
better before correlating, so a positive rho always means agreement.

The verdict has three branches, checked in this order:

* ``weak``: rho < STRONG_RHO. Rewrite the surrogate; no sampler can fix a model that ranks
  solutions badly. A weak model that also has ties at its optimum is still reported as weak:
  breaking ties does not repair the ranking.
* ``underdetermined``: the ranking is strong, but at least ``min_ties`` solutions tie at the
  model's optimum and the referee scores them differently, so a sampler is drawing lots among
  them. Add a tie-breaking term.
* ``strong``: invest in the sampler, depth and partitions.

Nothing here knows about any particular problem: it takes two arrays over the same solutions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

STRONG_RHO = 0.7
"""Rank correlation above which the surrogate is worth investing in.

Statistical significance is NOT this threshold and must not be confused with it: with a few
hundred solutions, rho = 0.14 already clears p < 0.05 (the critical value is about
1.96/sqrt(n-1), so 0.134 at n = 216, the number of valid network05 topologies) while being far
too weak to act on. That is why ``significant`` and ``strong`` are separate fields.
"""

MIN_TIES = 20
"""Ties at the model's optimum that make it underdetermined: a handful is noise, a plateau of
tens of solutions means the model cannot choose."""


@dataclass(frozen=True)
class Agreement:
    """How well a model's energies reproduce the referee's ranking."""

    rho: float                  # Spearman, already oriented: positive always means agreement
    p_value: float
    n: int
    significant: bool           # rho > 0 and p < 0.05: distinguishable from no agreement at all
    strong: bool                # rho >= STRONG_RHO: strong enough to invest in the sampler
    rank_of_best: int           # model's rank of the referee's best solution (1 = best), optimistic end
    n_tied_with_best: int       # model energies equal to it: the rank is a range this wide
    n_referee_optima: int       # solutions sharing the referee's best score
    n_tied_at_model_optimum: int
    referee_spread_at_model_optimum: float   # worst - best referee score among those ties
    referee_best: float
    referee_at_model_optimum: float          # best referee score among the ties: optimistic
    referee_mean_at_model_optimum: float     # what picking at random among them actually gives
    min_ties: int = MIN_TIES

    @property
    def degenerate(self) -> bool:
        """Many solutions tie at the model's optimum and the referee tells them apart."""
        return self.n_tied_at_model_optimum >= self.min_ties and self.referee_spread_at_model_optimum > 0

    @property
    def verdict(self) -> str:
        """``weak``, ``underdetermined`` or ``strong``: see the module docstring."""
        if not self.strong:
            return "weak"
        if self.degenerate:
            return "underdetermined"
        return "strong"

    def __str__(self) -> str:
        advice = {
            "strong": f"the model ranks like the referee (rho >= {STRONG_RHO}): invest in sampler, depth, partitions",
            "weak": f"AGREEMENT TOO WEAK to act on (needs rho >= {STRONG_RHO}): rewrite the surrogate",
            "underdetermined": "MODEL UNDERDETERMINED: the sampler draws lots among its optima; add a tie-breaking term",
        }[self.verdict]
        sig = "distinguishable from zero" if self.significant else "not distinguishable from zero"
        rank = f"{self.rank_of_best}"
        if self.n_tied_with_best > 1:
            rank += f"-{self.rank_of_best + self.n_tied_with_best - 1} (tied)"
        lines = [
            f"rho (oriented: positive = agreement) = {self.rho:+.3f}  (p = {self.p_value:.3g}, n = {self.n}, {sig})",
            f"  -> {advice}",
            f"model's rank of the referee's best solution: {rank} / {self.n}"
            + (f"  (best-placed of {self.n_referee_optima} referee optima)" if self.n_referee_optima > 1 else ""),
            f"solutions tied at the model's optimum: {self.n_tied_at_model_optimum}",
            f"  referee score on those: best {self.referee_at_model_optimum:.6g}, "
            f"mean {self.referee_mean_at_model_optimum:.6g}, spread {self.referee_spread_at_model_optimum:.6g}"
            f"  (overall best {self.referee_best:.6g})",
        ]
        if self.verdict == "weak" and self.degenerate:
            lines.append(f"  -> also {self.n_tied_at_model_optimum} ties at the optimum; fix the ranking first")
        return "\n".join(lines)


def rank_agreement(model_energy, referee_score, *,
                   model_lower_is_better: bool,
                   referee_lower_is_better: bool,
                   tolerance: float = 0.0,
                   min_ties: int = MIN_TIES) -> Agreement:
    """Spearman agreement between a model and the referee, oriented so positive means agreement.

    ``model_energy`` and ``referee_score`` are values over the SAME solutions, in the same order.
    Both polarity flags are required: see the module docstring.

    ``tolerance`` widens what counts as a tie in the model's energies, for energies that carry
    floating-point noise. Referee scores are compared exactly (QOBLIB scores are integers).
    """
    model = np.asarray(model_energy, dtype=float)
    referee = np.asarray(referee_score, dtype=float)
    if model.shape != referee.shape or model.ndim != 1:
        raise ValueError(f"expected two 1-D arrays of the same length, got {model.shape} and {referee.shape}")
    if len(model) < 3:
        raise ValueError(f"need at least 3 solutions to rank, got {len(model)}")
    if np.isnan(model).any() or np.isnan(referee).any():
        # A NaN ranks nowhere and would silently empty the set of ties at the optimum.
        raise ValueError("NaN in the inputs: a surrogate or a score failed to evaluate")

    # Orient both so that SMALLER is better, then agreement is a positive correlation.
    m = model if model_lower_is_better else -model
    r = referee if referee_lower_is_better else -referee

    result = spearmanr(m, r)
    rho, p = float(result.statistic), float(result.pvalue)

    # The referee's optimum may be shared. Rank the best-placed of those solutions: that is the
    # one a sampler walking down the model's ranking meets first, and the choice no longer
    # depends on the order in which the solutions were listed.
    referee_optima = np.flatnonzero(r == r.min())
    best_referee = int(referee_optima[np.argmin(m[referee_optima])])
    rank_of_best = int((m < m[best_referee] - tolerance).sum()) + 1
    n_tied_with_best = int((np.abs(m - m[best_referee]) <= tolerance).sum())

    tied = m <= m.min() + tolerance
    spread = float(r[tied].max() - r[tied].min())

    # Report the referee's own units, not the oriented ones. `r` is oriented so that smaller is
    # better whatever the referee's convention: the best tie is r.min(), and `sign` puts it back.
    sign = 1.0 if referee_lower_is_better else -1.0
    return Agreement(
        rho=rho,
        p_value=p,
        n=len(model),
        significant=bool(rho > 0 and p < 0.05),
        strong=bool(rho >= STRONG_RHO),
        rank_of_best=rank_of_best,
        n_tied_with_best=n_tied_with_best,
        n_referee_optima=len(referee_optima),
        n_tied_at_model_optimum=int(tied.sum()),
        referee_spread_at_model_optimum=spread,
        referee_best=float(sign * r.min()),
        referee_at_model_optimum=float(sign * r[tied].min()),
        referee_mean_at_model_optimum=float(sign * r[tied].mean()),
        min_ties=min_ties,
    )


def baseline_table(**named_scores) -> str:
    """One line per method, best first, for keeping the classical bar next to every run.

    Every value is a referee score; lower is better for both QOBLIB problems. A quantum result
    without the classical bar beside it says nothing about the quantum step.
    """
    if not named_scores:
        raise ValueError("no scores given")

    rows = sorted(named_scores.items(), key=lambda kv: kv[1])
    width = max(len(name) for name in named_scores)
    best = rows[0][1]

    def gap(score: float) -> str:
        if score == best:
            return ""
        return f"   {(score - best) / abs(best):+.2%}" if best != 0 else f"   {score - best:+.6g}"

    return "\n".join(f"{name:<{width}}  {score:>14.6g}" + gap(score) for name, score in rows)
