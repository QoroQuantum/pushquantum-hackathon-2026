"""Compatibility entry point for parallel tempering."""

import sys

from network_design import parallel_tempering

if __name__ == "__main__":
    parallel_tempering.main()
else:
    sys.modules[__name__] = parallel_tempering
