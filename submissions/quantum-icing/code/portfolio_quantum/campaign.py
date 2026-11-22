"""Run a list of QLNS jobs with a RAM guard (the machine has 7.4 GB; Pylance takes 1-2 GB).

Each job line: <inst> <lam> <start> <minutes> [extra qlns.py args...]
Usage: python portfolio_quantum/campaign.py jobs.txt --parallel 3 --logdir logs/
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def mem_available_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def qlns_processes() -> int:
    """QLNS python processes running on the machine (ours or started elsewhere)."""
    n = 0
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                cmd = (d / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if cmd and b"python" in cmd[0] and any(c.endswith(b"qlns.py") for c in cmd):
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--logdir", default=str(HERE / "logs"))
    ap.add_argument("--min-mem", type=int, default=450)
    a = ap.parse_args()
    logdir = Path(a.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    jobs = [l.split() for l in Path(a.jobs).read_text().splitlines() if l.strip() and not l.startswith("#")]
    running = []
    while jobs or running:
        running = [(p, n, j) for p, n, j in running if p.poll() is None]
        if jobs and qlns_processes() < a.parallel and mem_available_mb() > a.min_mem + 600:
            job = jobs.pop(0)
            inst, lam, start, minutes, *extra = job
            name = f"{inst}_l{lam}_{Path(start).stem if '/' in start else start}_{int(time.time()) % 100000}"
            cmd = [sys.executable, str(HERE / "qlns.py"), "--inst", inst, "--lam", lam, "--start", start,
                   "--minutes", minutes, "--solver", "qaoa", *extra]
            log = open(logdir / f"{name}.log", "w")
            log.write(" ".join(cmd) + "\n")
            log.flush()
            running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), name, job))
            print(time.strftime("%H:%M:%S"), "started", name, flush=True)
            time.sleep(20)
            continue
        if mem_available_mb() < a.min_mem and running:  # guard: stop the newest job, re-queue it
            p, n, j = running.pop()
            p.terminate()
            jobs.insert(0, j)
            print(time.strftime("%H:%M:%S"), "memory low, stopped and re-queued", n, flush=True)
            time.sleep(60)
        time.sleep(5)
    print("campaign done", flush=True)


if __name__ == "__main__":
    main()
