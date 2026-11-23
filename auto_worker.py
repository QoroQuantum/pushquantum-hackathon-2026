#!/usr/bin/env python
"""``auto_runner.py`` worker for our pipeline (SA start -> asset-group blocks -> iterated search).

    python auto_runner.py auto_worker.py --tests local_size_tests --silent -- --time 300
    python auto_runner.py auto_worker.py --tests local_size_tests --silent \
        -- --solver qaoa --size 4 --time 600   # QAOA blocks, TQA start

The runner calls this once per instance from its run directory with

    --instance po_a050_t10_orig_b020_l1e-3 --instance-dir <rel. path to instances/po_a050_t10_orig>
    --assets 50 --days 10 --lambda 1e-3 --budget 20 --cloud false [--solver-seed N]

and we write, in the working directory, ``<instance>.sol``, ``pipeline_*.json`` and
``<instance>.result.json`` (``{"score", "best_sol", "extra"}``). The default
``--solver exact`` is classical; ``--solver qaoa`` (TQA start) runs its blocks on the
local Maestro simulator, so ``--cloud`` is accepted and ignored. If ``--data-dir`` is
missing, the instance is downloaded from QOBLIB. ``--postprocess`` does nothing (it is
there so the runner's post-processing call succeeds).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import qoblib_tools as qt  # noqa: E402
from asset_groups import add_solver_args, run_pipeline  # noqa: E402


def load_instance_dir(path: Path, name: str) -> qt.PortfolioInstance:
    """Parse ``stock_prices.txt.gz`` + ``covariance_matrices.txt.gz`` like ``qt.load_portfolio_instance``."""
    symbols, index, entries = [], {}, []
    for day, sym, price in qt._data_rows(path / "stock_prices.txt.gz"):
        if sym not in index:
            index[sym] = len(symbols)
            symbols.append(sym)
        entries.append((index[sym], int(day), Fraction(price)))
    n_periods = max(t for _, t, _ in entries) + 1
    prices = [[None] * n_periods for _ in symbols]
    for i, t, p in entries:
        prices[i][t] = p
    if any(p is None for row in prices for p in row):
        raise ValueError(f"{name}: missing prices")
    cov = {}
    for day, s1, s2, value in qt._data_rows(path / "covariance_matrices.txt.gz"):
        t = int(day)
        if s1 in index and s2 in index and t < n_periods:
            cov[index[s1], index[s2], t] = Fraction(value)
    return qt.PortfolioInstance(name, symbols, n_periods, prices, cov)


def best_known(po_base: str, budget: int, lam: str):
    try:
        header, _ = qt.read_portfolio_solution(qt.fetch_portfolio_reference(po_base, budget=budget, lam_tag=lam))
        return int(header["objective"])
    except Exception:  # a200/a400 have none, or no network
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--postprocess", nargs=2, metavar=("RUN_DIR", "RUN_JSON"))
    p.add_argument("--instance")
    p.add_argument("--instance-dir", "--data-dir", dest="data_dir")
    p.add_argument("--assets", type=int)
    p.add_argument("--days", type=int)
    p.add_argument("--lambda", "--lam", dest="lam")
    p.add_argument("--budget", type=int)
    p.add_argument("--cloud", default="false",
                   help="ignored: QAOA blocks (--solver qaoa) run on the local Maestro simulator")
    # pipeline options (defaults as in `asset_groups.py pipeline`)
    p.add_argument("--size", type=int, default=5, help="asset-group size")
    p.add_argument("--method", default="contiguous",
                   choices=["contiguous", "random", "louvain", "spectral", "neighbourhood"])
    p.add_argument("--time", type=float, default=120.0,
                   help="iterated-search time limit (s); 0 = no clock: sweep blocks until converged, "
                        "then --kicks kicks")
    p.add_argument("--kicks", type=int, default=0, help="with --time 0: number of kicks after the sweeps")
    p.add_argument("--kick-blocks", type=int, default=6)
    p.add_argument("--kick-one-day", action="store_true", help="put all kicked blocks on one day")
    p.add_argument("--sa-reads", type=int, default=20)
    p.add_argument("--sa-sweeps", type=int, default=2000)
    start = p.add_mutually_exclusive_group()
    start.add_argument("--no-sa", action="store_true", help="start from an empty portfolio (use on a400)")
    start.add_argument("--init", help="start from a feasible .sol file instead of whole-QUBO SA")
    p.add_argument("--seed", type=int, default=1909)
    p.add_argument("--solver-seed", type=int, help="seed supplied by AutoRunner (overrides --seed)")

    add_solver_args(p)
    args = p.parse_args(argv)
    if args.solver_seed is not None:
        args.seed = args.solver_seed
    if not 0 < args.day_tscale < float("inf"):
        p.error("--day-tscale must be finite and positive")
    if args.postprocess:
        print("auto_worker: no post-processing")
        return 0
    missing = [k for k in ("instance", "lam", "budget") if getattr(args, k) is None]
    if missing:
        p.error(f"missing arguments: {missing}")

    m = re.match(r"^(po_a\d+_t\d+_(?:orig|s\d\d))_b\d+_l.+$", args.instance)
    if not m:
        raise SystemExit(f"cannot parse --instance {args.instance!r}")
    po_base = m.group(1)
    data = Path(args.data_dir) if args.data_dir else None
    inst = load_instance_dir(data, po_base) if data and data.is_dir() else qt.load_portfolio_instance(po_base)
    if args.assets is not None and (len(inst.symbols), inst.n_periods) != (args.assets, args.days):
        print(f"WARNING: data has {len(inst.symbols)} assets x {inst.n_periods} days, "
              f"expected {args.assets} x {args.days}", flush=True)
    model = qt.PortfolioModel(inst, budget=args.budget, lam=args.lam)
    bkv = best_known(po_base, args.budget, args.lam)
    print(f"{args.instance}: {len(inst.symbols)} assets, {inst.n_periods} days; best known {bkv}", flush=True)

<<<<<<< HEAD
    pipe_args = argparse.Namespace(lam=args.lam, size=args.size, methods=[args.method], time=args.time,
                                   kick_blocks=args.kick_blocks, kicks=args.kicks, sa_reads=args.sa_reads,
                                   sa_sweeps=args.sa_sweeps, no_sa=args.no_sa, seed=args.seed,
                                   solutions=".", out=".", solver=args.solver,
                                   qaoa_layers=args.qaoa_layers, qaoa_iters=args.qaoa_iters,
                                   qaoa_shots=args.qaoa_shots, qaoa_top=args.qaoa_top,
                                   qaoa_init=args.qaoa_init, tqa_dt=args.tqa_dt,
                                   qaoa_compare_exact=args.qaoa_compare_exact,
                                   qaoa_optimizer=args.qaoa_optimizer, qaoa_population=args.qaoa_population,
                                   tqa_dt_range=args.tqa_dt_range, tqa_dt_sampling=args.tqa_dt_sampling,
                                   qaoa_backend=args.qaoa_backend,
                                   day_blocks=args.day_blocks, day_chains=args.day_chains,
                                   day_sweeps=args.day_sweeps, day_share=args.day_share)
=======
    pipe_args = argparse.Namespace(**vars(args), methods=[args.method], solutions=".", out=".")
>>>>>>> 6784ddb605f5e27c6b76dfaf32d6f6393abe1352
    rec = run_pipeline(model, bkv, pipe_args)
    rec["kick_one_day"] = args.kick_one_day
    sol = Path(rec["solution_file"])
    if sol.name != f"{args.instance}.sol":  # run_pipeline names it <po_base>_bBBB_l<lam>.sol
        sol = sol.rename(f"{args.instance}.sol")
    rec.pop("trace", None)
    payload = {"score": rec["score"], "best_sol": sol.name, "extra": rec}
    Path(f"{args.instance}.result.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print("AUTORUNNER_RESULT: " + json.dumps(payload, default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
