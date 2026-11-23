"""Upload staged a010 QAOA solutions to your aqora store (then press Submit on the aqora page).

Run with the aqora CLI's Python (default: the λ1e-3 file):
    ~/.local/share/uv/tools/aqora/bin/python submission/upload.py
    ~/.local/share/uv/tools/aqora/bin/python submission/upload.py po_a010_t10_orig_b004_l1e-4
"""
import sys
from pathlib import Path

import aqora

here = Path(__file__).parent
store = aqora.Store()
for name in sys.argv[1:] or ["po_a010_t10_orig_b004_l1e-3"]:
    store.put(f"pushquantum-2026-qoro/{name}.sol", (here / f"{name}.sol").read_bytes())
    store.put(
        f"pushquantum-2026-qoro/{name}.json",
        (here / f"{name}.json").read_bytes(),
        content_type="application/json",
    )
    print("uploaded", name, "->", store.get(f"pushquantum-2026-qoro/{name}.json").decode())
