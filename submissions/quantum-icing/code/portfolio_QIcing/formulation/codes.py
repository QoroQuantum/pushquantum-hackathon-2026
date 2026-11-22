"""PCE parity encodings seen as linear codes, plus problem-aware variants.

Divi's parity encodings (DenseEncoding, PolyEncoding) set variable i to 1 where the Z-parity of its
qubit mask is even:  x = 1 XOR (M z)  over GF(2), z the measured bitstring.  Every sampled shot is
therefore a codeword of an affine code of dimension q (2^q words among 2^N assignments).

Consequence for sparse problems (budget sum|v| <= B):  in PolyEncoding a pair variable equals
[z_a == z_b], so if k of the q single qubits read 1, C(k,2) + C(q-k,2) >= ~q^2/4 ~ N/2 pair variables
are 1 in every shot.  Sampled portfolios are dense and almost never feasible -> the greedy repair does
the work.

Fixes implemented here:
  * reference offset ("re-centred code"): complement the polynomial on the variables where the
    reference x_ref is 0, so the decoded problem variable is  x_ref XOR (M z).  z = 0 -> x_ref
    (empty portfolio, or an incumbent), and the code is a structured neighbourhood of x_ref.
  * MaskEncoding: any list of qubit masks (Divi ParityEncoding subclass) -> custom codes.
  * structure-aware mask assignment: give the cheap-to-flip strings (single qubits) to the
    magnitude bits of the most promising assets, pair strings to sign/slack bits, where a dense
    pattern is harmless (sign is irrelevant at magnitude 0, slack is free).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from divi.qprog.algorithms._pce import ParityEncoding

from formulations import Poly


@dataclass(frozen=True)
class MaskEncoding(ParityEncoding):
    """Parity encoding with explicit masks: variable i = [parity of qubits masks[i] is even]."""

    masks: tuple = ()
    q: int = 0

    def min_qubits(self, n_vars: int) -> int:
        return self.q

    def strings(self, n_vars: int, n_qubits: int):
        assert n_vars == len(self.masks), (n_vars, len(self.masks))
        return [("Z" * len(m), tuple(sorted(m))) for m in self.masks]


def poly_masks(n_vars: int, q: int | None = None) -> tuple[tuple, int]:
    """Divi PolyEncoding's masks, in Divi's order (singles, then pairs)."""
    if q is None:
        q = (math.isqrt(8 * n_vars + 1) - 1) // 2
        q = q if q * (q + 1) // 2 >= n_vars else q + 1
    last = q - 1
    singles = [(last - i,) for i in range(q)]
    pairs = [(last - j, last - i) for i in range(q) for j in range(i + 1, q)]
    return tuple(itertools.islice(itertools.chain(singles, pairs), n_vars)), q


def dense_masks(n_vars: int) -> tuple[tuple, int]:
    q = n_vars.bit_length()
    return tuple(tuple(sorted(q - 1 - j for j in range(q) if ((i + 1) >> j) & 1)) for i in range(n_vars)), q


def structured_poly_masks(priority: np.ndarray, q: int | None = None) -> tuple[tuple, int]:
    """Poly strings, but variables in `priority` order (highest first) get singles, then pairs.

    priority[i]: larger = more likely to be 1 in a good solution / should be cheap to flip alone.
    """
    n = len(priority)
    masks, q = poly_masks(n, q)
    order = np.argsort(-np.asarray(priority), kind="stable")
    out = [None] * n
    for slot, var in enumerate(order):
        out[var] = masks[slot]
    return tuple(out), q


def mask_matrix(masks, q) -> np.ndarray:
    M = np.zeros((len(masks), q), dtype=np.uint8)
    for i, m in enumerate(masks):
        M[i, list(m)] = 1
    return M


def codewords(masks, q, flip: np.ndarray | None = None, limit: int = 1 << 22) -> np.ndarray:
    """All 2^q decoded assignments (Divi convention x = 1 XOR Mz, then XOR flip). Shape (2^q, N)."""
    if (1 << q) > limit:
        raise ValueError(f"2^{q} codewords is too many to enumerate")
    M = mask_matrix(masks, q)
    Z = ((np.arange(1 << q)[:, None] >> np.arange(q - 1, -1, -1)[None, :]) & 1).astype(np.uint8)  # qubit 0 = MSB
    X = 1 - ((Z @ M.T) & 1)
    if flip is not None:
        X ^= flip.astype(np.uint8)[None, :]
    return X.astype(np.uint8)


def complement(poly: Poly, flip: np.ndarray) -> Poly:
    """p'(y) = p(y XOR flip): substitute x_i = 1 - y_i on flipped variables (exact, multilinear)."""
    F = set(np.flatnonzero(flip).tolist())
    out = Poly(const=poly.const)
    t = out.t
    for k, c in poly.t.items():
        fixed = [i for i in k if i not in F]
        fl = [i for i in k if i in F]
        for r in range(len(fl) + 1):
            for S in itertools.combinations(fl, r):
                key = tuple(sorted(fixed + list(S)))
                coef = c * (-1) ** r
                if key:
                    t[key] = t.get(key, 0.0) + coef
                else:
                    out.const += coef
    return out.clean()


def reference_flip(x_ref: np.ndarray) -> np.ndarray:
    """Flip set so that the Divi codeword with z = 0 (all x_divi = 1) decodes to x_ref."""
    return (1 - np.asarray(x_ref)).astype(np.uint8)
