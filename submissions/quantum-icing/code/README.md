# Code

All source code, notebooks, run logs and supporting material. Not reorganised yet: the folder is the old repo root, moved here as is.

- Notebooks: `00_main_qubo_101.ipynb` (start here), `01_portfolio_optimisation.ipynb`, `02_network_design.ipynb`
- Network design: `network_*.py`, `network_design/`, `TheGOAT_Testing/`, `hopqubo_newTesting/`; notes in `NETWORK_DESIGN.md`
- Portfolio: `portfolio_search.py`, `portfolio_QIcing/`, `portfolio_quantum/`
- Shared: `qoblib_tools.py`, `tests/`; `qoblib_data/` is downloaded by the notebooks (git-ignored)
- `ai_context/`: methodology, logs and run evidence. `website/`: static site (deployed via `../render.yaml`)

Scripts with hard-coded paths such as `deliverables/` or `solutions/` still expect the old root layout. Run them from inside `code/`, or point them at `../all_results/...` with environment variables such as `NETWORK_SOL_DIR`.
