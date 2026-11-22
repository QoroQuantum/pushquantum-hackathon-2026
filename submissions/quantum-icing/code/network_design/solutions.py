"""Read and save checked incumbents using the starter kit's referee."""

from pathlib import Path
from tempfile import TemporaryDirectory

import qoblib_tools as qt


BEST_KNOWN = {11: 238_334, 24: 663_688}


def checked_score(path, demand):
    valid, score, message = qt.check_network_solution(path, len(demand), demand)
    if not valid:
        raise RuntimeError(f"{path}: {message}")
    return score


def save_checked_solution(path, demand, routing, expected_score=None):
    """Check the serialized solution before saving a new best at this path."""
    n_nodes = len(demand)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as staging_dir:
        candidate = Path(staging_dir) / path.name
        written_score = qt.write_network_solution(candidate, n_nodes, routing)
        score = checked_score(candidate, demand)
        if score != written_score:
            raise RuntimeError(f"{path.stem} scored {score:,}; file claims {written_score:,}")
        if expected_score is not None and score != expected_score:
            raise RuntimeError(f"{path.stem} scored {score:,}; expected {expected_score:,}")

        if path.is_file():
            valid, saved_score, _ = qt.check_network_solution(path, n_nodes, demand)
            if valid and saved_score < score:
                return saved_score
        candidate.replace(path)
    return score
