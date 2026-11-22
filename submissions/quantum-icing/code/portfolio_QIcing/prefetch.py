"""Download instance data sequentially (avoids races between parallel workers)."""
import pf_core  # noqa: F401  (sets sys.path)
import qoblib_tools as qt

names = [f"po_a050_t{T}_{s}" for T in (10, 15) for s in ("orig", "s00", "s01", "s02")]
names += ["po_a200_t10_orig", "po_a400_t10_orig"]
for n in names:
    b = f"06-portfolio/instances/{n}"
    qt.fetch_qoblib_file(b + "/stock_prices.txt.gz")
    qt.fetch_qoblib_file(b + "/covariance_matrices.txt.gz")
    print("ok", n, flush=True)
