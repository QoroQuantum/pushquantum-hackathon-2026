"""Compatibility entry point for the shared network pipeline."""

import sys

from network_design import pipeline

if __name__ == "__main__":
    pipeline.main()
else:
    sys.modules[__name__] = pipeline
