"""Render the slide deck's own files to a PDF, one 1920x1080 page per slide.

The deck lives as an Artifact (private, and the Share menu exports it), but the hand-in needs a
file anyone on the team can open from the repository. The deck's content is plain HTML and CSS,
so this stitches the slides into one standalone page and prints it with headless Chrome.

Input is a folder holding the deck's own files at their published paths:

    <src>/project/deck.json          the index: `order` gives the slide order, `faces` the fonts
    <src>/project/deck.css           the shared rules
    <src>/project/slides/<id>.html   one <section> per slide

Speaker notes (`<aside>`) are dropped from the pages and written beside the PDF as a text file,
because the PDF is for the audience. Webfonts are fetched from Google Fonts when `faces` names
them; without a network the CSS falls back to Arial and the layout still holds.

    venv/bin/python TheGOAT_Testing/build_deck_pdf.py --src <folder> --out TheGOAT_Testing/deck.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PAGE_CSS = """
@page { size: 1920px 1080px; margin: 0; }
html, body { margin: 0; padding: 0; background: #fff; }
section {
  width: 1920px; height: 1080px; box-sizing: border-box;
  position: relative; overflow: hidden;
  break-after: page; page-break-after: always;
}
section:last-of-type { break-after: auto; page-break-after: auto; }
aside { display: none !important; }
"""


def chrome() -> str:
    for name in ("google-chrome", "chromium", "chromium-browser", "google-chrome-stable"):
        found = shutil.which(name)
        if found:
            return found
    sys.exit("no Chrome/Chromium on PATH: cannot render the deck")


def build_html(src: Path) -> tuple[str, str]:
    """(standalone HTML, speaker notes) from the deck's own files."""
    index = json.loads((src / "project" / "deck.json").read_text())
    order = index.get("order") or []
    css = (src / "project" / "deck.css").read_text() if (src / "project" / "deck.css").exists() else ""

    links = "".join(
        f'<link rel="stylesheet" href="{face["href"]}">'
        for face in (index.get("faces") or {}).values() if face.get("href")
    )

    sections, notes = [], []
    for slide_id in order:
        path = src / "project" / "slides" / f"{slide_id}.html"
        if not path.exists():
            print(f"  ! {slide_id}: no file, skipped")
            continue
        html = path.read_text()
        for note in re.findall(r"<aside>(.*?)</aside>", html, flags=re.S):
            notes.append(f"--- {slide_id} ---\n{re.sub(r'<[^>]+>', '', note).strip()}\n")
        sections.append(re.sub(r"<aside>.*?</aside>", "", html, flags=re.S).strip())

    title = index.get("title") or "deck"
    page = (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
            f'{links}<style>{css}\n{PAGE_CSS}</style></head><body>'
            + "\n".join(sections) + "</body></html>")
    return page, "\n".join(notes)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", required=True, help="folder holding project/deck.json and project/slides/")
    p.add_argument("--out", default="TheGOAT_Testing/deck.pdf")
    p.add_argument("--timeout", type=int, default=180)
    a = p.parse_args()

    src = Path(a.src).resolve()
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    page, notes = build_html(src)

    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "deck.html"
        html_path.write_text(page)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [chrome(), "--headless", "--disable-gpu", "--no-sandbox", "--virtual-time-budget=10000",
               "--no-pdf-header-footer", f"--print-to-pdf={out}", html_path.as_uri()]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        if not out.exists():
            sys.exit(f"Chrome produced no PDF (exit {r.returncode}):\n{r.stderr[-2000:]}")

    if notes:
        out.with_suffix(".notes.txt").write_text(notes)
    pages = len(re.findall(r"<section", page))
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB, {pages} slides)")
    if notes:
        print(f"wrote {out.with_suffix('.notes.txt').relative_to(ROOT)} (speaker notes)")


if __name__ == "__main__":
    main()
