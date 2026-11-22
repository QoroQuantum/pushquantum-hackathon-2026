"""Command-line entry point for the QOBLIB network search."""

from network_design.search import main, search, switch_links
from network_design.solutions import checked_score


if __name__ == "__main__":
    main()
