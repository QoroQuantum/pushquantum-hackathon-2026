#!/usr/bin/env python
"""Random vs TQA initialisation on po_a010_t10_orig_b004 (budget 4), p = 3, λ = 1e-6, 1e-4, 1e-2, 3 seeds.

    python tqa_vs_random_a010.py            # results in AutoData/po_a010_t10_orig_b004_p3/
    python plot_tqa_vs_random.py AutoData/po_a010_t10_orig_b004_p3   # replot only

Extra options are passed through, e.g. --workers 3 (a later --lams replaces these).
"""
import sys

import tqa_vs_random

ARGS = ["--instance", "po_a010_t10_orig_b004", "--layers", "3", "--seeds", "1", "2", "3",
        "--lams", "1e-6", "1e-4", "1e-2"]

if __name__ == "__main__":
    sys.exit(tqa_vs_random.main(ARGS + sys.argv[1:]))
