"""
A file for sending the solution to Aqora: run this file, with the file name as a command line argument. 
This only queues the project, you still have to submit on Aqora website. 
note that you must 
pip install aqora and also run: "aqora login" command.
"""


import argparse
import json
from pathlib import Path

import aqora

parser = argparse.ArgumentParser(description="Send a solution file to Aqora.")
parser.add_argument(
    "filename",
    help="Solution file name (with or without the .sol extension).",
)
args = parser.parse_args()

raw_name = args.filename
sol_path = Path(raw_name if raw_name.endswith(".sol") else f"{raw_name}.sol")
if not sol_path.is_file():
    raise FileNotFoundError(f"Solution file not found: {sol_path}")

name = sol_path.name.removesuffix(".sol")

store = aqora.Store()
store.put(f"pushquantum-2026-qoro/{name}.sol", sol_path.read_bytes())
store.put(
    f"pushquantum-2026-qoro/{name}.json",
    json.dumps({"qubits": 12, "method": "bringing the quantum heat!"}).encode(),
    content_type="application/json",
)