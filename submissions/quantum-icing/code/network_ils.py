"""Compatibility entry point for iterated local search."""

import sys

from network_design import iterated_local_search

if __name__ == "__main__":
    iterated_local_search.main()
else:
    sys.modules[__name__] = iterated_local_search
