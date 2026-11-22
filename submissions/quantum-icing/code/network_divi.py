"""Compatibility entry point for the Divi swap-selection experiment."""

import sys

from network_design import divi_proposer

if __name__ == "__main__":
    divi_proposer.main()
else:
    sys.modules[__name__] = divi_proposer
