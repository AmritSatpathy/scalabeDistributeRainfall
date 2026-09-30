"""
Build a self-contained HTML (images embedded as base64) from report/REPORT.md,
which any browser can "Print -> Save as PDF" to produce the submission PDF.

This avoids a heavy LaTeX install. If `pandoc` + a PDF engine are available,
producing a PDF directly is also fine (see README); this script is the
zero-heavy-dependency fallback.

Usage:
    pip install markdown
    python report/build_pdf.py
    # then open report/REPORT.html in a browser and Print -> Save as PDF
"""
from __future__ import annotations

import base64
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(HERE, "REPORT.md")
HTML_PATH = os.path.join(HERE, "REPORT.html")

CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       max-width: 900px; margin: 2em auto; line-height: 1.5; color: #1a1a1a;
       padding: 0 1em; }
h1 { font-size: 1.8em; border-bottom: 2px solid #333; padding-bottom: .3em; }
h2 { font-size: 1.4em; border-bottom: 1px solid #ccc; padding-bottom: .2em; margin-top: 1.6em; }
h3 { font-size: 1.15em; margin-top: 1.2em; }
table { border-collapse: collapse; margin: 1em 0; font-size: .95em; }
th, td { border: 1px solid #bbb; padding: 4px 10px; text-align: left; }
th { background: #f0f0f0; }
img { max-width: 100%; height: auto; display: block; margin: 1em auto;
      border: 1px solid #ddd; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
pre { background: #f4f4f4; padding: 1em; overflow-x: auto; border-radius: 4px; }
blockquote { border-left: 4px solid #bbb; margin: 1em 0; padding: .3em 1em; color: #444; background:#fafafa; }
"""


def embed_images(md: str) -> str:
    """Replace ![alt](path) with base64 data URIs so the HTML is self-contained."""
    def repl(m):
        alt, path = m.group(1), m.group(2)
        abspath = os.path.normpath(os.path.join(HERE, path))
        if not os.path.exists(abspath):
            return m.group(0)
        with open(abspath, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return f'![{alt}](data:image/png;base64,{b64})'
    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, md)


def main() -> None:
    try:
        import markdown  # type: ignore
    except ImportError:
        raise SystemExit("Please: pip install markdown")

    with open(MD_PATH, encoding="utf-8") as fh:
        md = fh.read()
    md = embed_images(md)
    html_body = markdown.markdown(
        md, extensions=["tables", "fenced_code", "toc"]
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{html_body}</body></html>"
    )
    with open(HTML_PATH, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {HTML_PATH}")
    print("Open it in a browser and use Print -> Save as PDF for the submission PDF.")


if __name__ == "__main__":
    main()
