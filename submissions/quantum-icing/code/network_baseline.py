"""Command-line entry point for the QOBLIB network baseline."""

from network_design.baseline import NETWORK05_OPTIMUM, OUTPUT_DIR, main
from network_design.solutions import BEST_KNOWN, save_checked_solution


NETWORK24_BEST_KNOWN = BEST_KNOWN[24]


if __name__ == "__main__":
    main()
