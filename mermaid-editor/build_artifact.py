#!/usr/bin/env python3
"""Produce a body-only copy of the standalone editor for hosting on a platform
that supplies its own <!doctype>/<html>/<head>/<body> wrapper.

Not part of the offline editor — mermaid-editor.html is what users open
locally. Run after build_single_file.py when refreshing the hosted copy.
"""

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
SRC = HERE / "mermaid-editor.html"
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "hosted-body.html"


def main() -> None:
    if not SRC.exists():
        sys.exit("ERROR: run build_single_file.py first")
    html = SRC.read_text(encoding="utf-8")

    # The vendored mermaid bundle contains literal "</head>" and "</body>"
    # strings, so slice on the outermost tags rather than regex-matching.
    def slice_between(open_tag: str, close_tag: str) -> str:
        start = html.find(open_tag)
        end = html.rfind(close_tag)
        if start == -1 or end == -1 or end < start:
            sys.exit(f"ERROR: could not locate {open_tag}…{close_tag}")
        return html[start + len(open_tag):end]

    head = slice_between("<head>", "</head>")
    body = slice_between("<body>", "</body>")

    # Keep <title> and <style> from the head; drop <meta>/<link>, which the
    # host wrapper provides.
    keep = []
    title = re.search(r"<title>.*?</title>", head, re.S)
    if title:
        keep.append(title.group(0))
    keep += re.findall(r"<style>.*?</style>", head, re.S)

    out = "\n".join(keep) + "\n" + body.strip() + "\n"
    OUT.write_text(out, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()
