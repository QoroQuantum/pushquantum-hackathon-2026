"""Alternate parallel tempering and ILS on one network instance, forever, until you stop it.

    python network_alternate.py                          # network24, output network24_best.sol
    python network_alternate.py --out my24.sol --workers 3 --pt-minutes 10 --ils-minutes 8

Each round runs parallel tempering (``network_pt.py``) and then ILS with the deep step
(``network_ils.py``). Both start from the best topology found so far and bank into the same work
folder. While they run, this script checks that folder every ``--poll`` seconds. When a strictly
better file appears, it re-checks it with the referee (``qoblib_tools.check_network_solution``) and
atomically overwrites ``--out``. So ``--out`` always holds the best checked solution, and you can
stop the script at any moment with Ctrl+C.

Restarting is safe: the script starts from the best of ``--start``, ``--out`` and the work folder.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import qoblib_tools as qt
from network_design.pipeline import BKV

REPO = Path(__file__).resolve().parent
STOP = False


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S}  {msg}", flush=True)


def checked(path: Path, n: int, demand) -> int | None:
    """Referee score of a .sol file, or None if it is missing or invalid."""
    if not path.exists():
        return None
    try:
        valid, score, _ = qt.check_network_solution(path, n, demand)
    except Exception:  # half-written or malformed file
        return None
    return score if valid else None


def atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)  # readers see either the old file or the new one, never half of it


def gap(n: int, score: int) -> str:
    return f"{(score - BKV[n]) / abs(BKV[n]):+.3%}"


class Keeper:
    """Tracks the best checked file and publishes it to ``out``."""

    def __init__(self, n: int, demand, work: Path, out: Path):
        self.n, self.demand, self.work, self.out = n, demand, work, out
        self.banked = work / f"network{n:02d}.sol"
        self.best = checked(out, n, demand)
        self.seen_mtime = None
        self.history = work / "improvements.csv"

    def offer(self, path: Path, source: str) -> bool:
        score = checked(path, self.n, self.demand)
        if score is None or (self.best is not None and score >= self.best):
            return False
        atomic_copy(path, self.out)
        prev, self.best = self.best, score
        new = not self.history.exists()
        with self.history.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "score", "gap", "previous", "source"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), score, gap(self.n, score), prev, source])
        log(f"NEW BEST {score:,} (gap {gap(self.n, score)}, was {prev if prev is None else f'{prev:,}'}) "
            f"from {source} → {self.out}")
        return True

    def poll(self, source: str) -> None:
        try:
            mtime = self.banked.stat().st_mtime_ns
        except FileNotFoundError:
            return
        if mtime != self.seen_mtime:
            self.seen_mtime = mtime
            self.offer(self.banked, source)

    def seed_workdir(self) -> None:
        """Make the work folder's banked file the best checked topology we have."""
        work_score = checked(self.banked, self.n, self.demand)
        if self.best is not None and (work_score is None or self.best < work_score):
            atomic_copy(self.out, self.banked)


def run_phase(cmd: list[str], env: dict, logfile: Path, keeper: Keeper, source: str, poll: float) -> None:
    with logfile.open("w") as fh:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)  # own process group: Ctrl+C reaches only us
        try:
            while proc.poll() is None:
                keeper.poll(source)
                if STOP:
                    break
                time.sleep(poll)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            keeper.poll(source)
    if proc.returncode not in (0, -signal.SIGTERM, -signal.SIGKILL) and not STOP:
        log(f"warning: {source} exited with code {proc.returncode}, see {logfile}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24, choices=sorted(BKV))
    ap.add_argument("--out", type=Path, default=None, help="output .sol, overwritten on every improvement "
                    "(default networkNN_best.sol in the repo root)")
    ap.add_argument("--start", type=Path, default=None, help="starting .sol (default "
                    "deliverables/networkNN/networkNN.sol if it exists)")
    ap.add_argument("--workdir", type=Path, default=None, help="default runs/alternate_networkNN")
    ap.add_argument("--workers", type=int, default=4, help="PT workers and ILS chains (max 5)")
    ap.add_argument("--pt-minutes", type=float, default=15)
    ap.add_argument("--ils-minutes", type=float, default=10)
    ap.add_argument("--rounds", type=int, default=0, help="0 = run until stopped")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--poll", type=float, default=2.0, help="seconds between checks for a new best")
    args = ap.parse_args()

    n = args.n
    out = (args.out or REPO / f"network{n:02d}_best.sol").resolve()
    work = (args.workdir or REPO / "runs" / f"alternate_network{n:02d}").resolve()
    start = args.start or REPO / "deliverables" / f"network{n:02d}" / f"network{n:02d}.sol"
    workers = max(1, min(args.workers, 5))
    (work / "logs").mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    demand = qt.load_network_demand(n)
    keeper = Keeper(n, demand, work, out)
    if start.exists():
        keeper.offer(start.resolve(), f"start file {start}")
    keeper.offer(keeper.banked, "previous work folder")
    keeper.seed_workdir()
    keeper.seen_mtime = keeper.banked.stat().st_mtime_ns if keeper.banked.exists() else None
    if keeper.best is None:
        log("no starting solution: the first round starts from random topologies")
    log(f"network{n:02d}: best {keeper.best if keeper.best is None else f'{keeper.best:,}'}"
        f"{'' if keeper.best is None else f' (gap {gap(n, keeper.best)})'} · output {out} · work {work}")
    log(f"each round: PT {args.pt_minutes:g} min + ILS {args.ils_minutes:g} min, {workers} workers. "
        "Ctrl+C to stop.")

    def stop(signum, _frame):
        global STOP
        if not STOP:
            log("stopping: ending the current search and saving the best result...")
        STOP = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    env = dict(os.environ, NETWORK_SOL_DIR=str(work), PYTHONUNBUFFERED="1")
    py = sys.executable
    # Continue the round numbering of earlier runs in this folder: new seeds, no overwritten logs.
    r = max((int(p.name[5:8]) for p in (work / "logs").glob("round[0-9][0-9][0-9]_*")), default=0)
    last = r + args.rounds
    while not STOP and (args.rounds == 0 or r < last):
        r += 1
        seed = args.seed + 100 * r
        pt = [py, "network_pt.py", str(n), "--minutes", str(args.pt_minutes), "--replicas", "10",
              "--workers", str(workers), "--tmin", "450", "--tmax", "2500", "--banked", "2",
              "--K", "60", "--p-double", "0.5", "--p-tri", "0.03", "--steps", "15",
              "--seed", str(seed), "--tag", f"alt_r{r:03d}"]
        log(f"round {r}: parallel tempering ({args.pt_minutes:g} min, seed {seed})")
        run_phase(pt, env, work / "logs" / f"round{r:03d}_pt.log", keeper, f"round {r} PT", args.poll)
        if STOP:
            break
        keeper.seed_workdir()
        deep_tol, kmax = ("0.015", "8") if r % 2 else ("0.02", "10")  # vary the ILS a little per round
        ils = [py, "network_ils.py", "run", "--n", str(n), "--chains", str(workers),
               "--minutes", str(args.ils_minutes), "--seed", str(seed + 50), "--deep", "12",
               "--deep_tol", deep_tol, "--kmax", kmax, "--starts", "banked", "--sync",
               "--log", str(work / "logs" / f"round{r:03d}_ils.log")]
        log(f"round {r}: ILS with deep step ({args.ils_minutes:g} min, seed {seed + 50})")
        run_phase(ils, env, work / "logs" / f"round{r:03d}_ils.out", keeper, f"round {r} ILS", args.poll)
        keeper.seed_workdir()
        log(f"round {r} done · best {keeper.best if keeper.best is None else f'{keeper.best:,}'}")

    final = checked(out, n, demand)
    log(f"stopped. {out} → " + (f"valid, score {final:,} (gap {gap(n, final)})" if final else "no valid file"))


if __name__ == "__main__":
    main()
