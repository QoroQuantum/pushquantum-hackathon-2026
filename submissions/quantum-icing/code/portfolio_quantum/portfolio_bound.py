"""A certified lower bound for the signed QOBLIB portfolio objective.

Use a tangent to a positive semidefinite shift of the covariance, and certify
the linear relaxation with exact integer arithmetic on a rounded dual solution.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

import qlns_core as qc
from trajectory_guide import trajectory_guide


def shifted_risk(p):
    """Certify positive semidefiniteness using an integer Gram matrix plus a diagonally dominant residual."""
    matrices = p.RS.copy()
    shifts = np.empty(p.T, dtype=np.int64)
    scale = 1 << 16
    for t, matrix in enumerate(matrices):
        min_eigenvalue = float(np.linalg.eigvalsh(matrix)[0])
        shift = max(0, math.ceil(-min_eigenvalue / 2)) + 1
        diagonal = np.diag_indices(p.n)
        matrix[diagonal] += 2 * shift
        factor = np.rint(scale * np.linalg.cholesky(matrix - np.eye(p.n))).astype(np.int64)
        if max(abs(int(matrix.min())), int(matrix.max())) * scale**2 >= 2**61:
            raise OverflowError("Risk matrix is too large for the certificate")
        if int(abs(factor).max())**2 * p.n >= 2**61:
            raise OverflowError("Gram matrix is too large for the certificate")
        residual = scale**2 * matrix - factor @ factor.T
        if np.any(2 * residual.diagonal() < np.abs(residual).sum(axis=1)):
            raise ArithmeticError(f"PSD certificate failed for period {t}")
        shifts[t] = shift
    return matrices, shifts


def lower_bound(p, anchor):
    hessian, shift = shifted_risk(p)
    count = p.T * p.n
    changes = count - p.n
    identity = sparse.eye(count, dtype=np.int64, format="csr")
    difference = (identity - sparse.diags(
        [np.ones(changes, dtype=np.int64)], [-p.n], shape=(count, count), dtype=np.int64
    ))[p.n:]
    diff_identity = sparse.eye(changes, dtype=np.int64, format="csr")
    sums = sparse.kron(
        sparse.eye(p.T, dtype=np.int64), np.ones((1, p.n), dtype=np.int64), format="csr"
    )
    # Variables are signed positions, absolute positions, short positions and absolute turnover.
    matrix = sparse.bmat([
        [identity, -identity, None, None],
        [-identity, -identity, None, None],
        [-identity, None, -identity, None],
        [difference, None, None, -diff_identity],
        [-difference, None, None, -diff_identity],
        [None, sums, None, None],
        [sums, None, None, None],
        [-sums, None, None, None],
    ], format="csr", dtype=np.int64)
    rhs = np.r_[
        np.zeros(3 * count + 2 * changes, dtype=np.int64),
        np.full(p.T, p.budget), np.full(p.T, p.C), np.full(p.T, p.maxcash - p.C),
    ]
    lower = np.r_[np.full(count, -p.ub), np.zeros(2 * count + changes, dtype=np.int64)]
    upper = np.r_[np.full(3 * count, p.ub), np.full(changes, 2 * p.ub)]

    # For every signed position x, x² <= ub * |x|. The tangent of H then
    # lower-bounds the original risk after the diagonal correction.
    scale = 1 << 10
    denom = 2 * scale**2
    z = np.rint(anchor * scale).astype(np.int64)
    hz = (hessian @ z[..., None]).squeeze(-1)
    interest = int(p.cashc[0] - p.cashc[1])
    if not np.array_equal(p.cashc, -interest * np.arange(p.maxcash + 1)):
        raise ValueError("The cash objective must be linear")
    coeff = np.r_[
        (denom * (p.lin + interest) + 2 * scale * hz).ravel(),
        (denom * (p.absc - p.ub * shift[:, None])).ravel(),
        denom * p.sc.ravel(),
        denom * p.w[1:].ravel(),
    ]
    offset = -int(np.sum(z * hz)) - denom * interest * p.C * p.T
    result = linprog(coeff / denom, A_ub=matrix, b_ub=rhs,
                     bounds=np.column_stack((lower, upper)), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    # A nonpositive inequality dual supplies a valid lower bound even when
    # rounded. Minimize the remaining linear form exactly over its box.
    dual = np.minimum(0, np.rint(denom * result.ineqlin.marginals)).astype(np.int64)
    residual = coeff - matrix.T @ dual
    box = int(residual @ np.where(residual >= 0, lower, upper))
    bound = int(dual @ rhs) + box + offset
    return -(-bound // denom), bound / denom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start", type=Path, help="Feasible canonical portfolio solution")
    args = parser.parse_args()
    p = qc.ps.Problem("po_a400_t15_orig", "1e-3")
    v = qc.load_v(p, args.start)
    if not p.feasible(v):
        parser.error("The start portfolio is infeasible")
    anchor = trajectory_guide(p, v, steps=700)
    bound, real_bound = lower_bound(p, anchor)
    print(f"Certified lower bound {bound} (scaled LP dual {real_bound:.6f})")
    print(f"Incumbent {p.objective(v)}. Maximum possible improvement {p.objective(v) - bound}.")


if __name__ == "__main__":
    main()
