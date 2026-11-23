import argparse
import qoblib_tools as qt


parser = argparse.ArgumentParser(description="Check a QOBLIB portfolio solution file.")
parser.add_argument(
    "filename",
    nargs="?",
    default="po_a010_t10_orig_b004_l1e-4.sol",
    help="Path to the .sol file (default: po_a010_t10_orig_b004_l1e-4.sol)",
)
args = parser.parse_args()
filename = args.filename

header, positions = qt.read_portfolio_solution(filename)
model = qt.PortfolioModel(
    qt.load_portfolio_instance(header["instance"]),
    budget=int(header["budget"]),
    lam=header["lambda"],
)
score = model.score(positions)
print("Your score: ", score)              # summary: objective, feasible, breakdown
print("Constraint violations: ", score.violations)   # empty list [] if feasible