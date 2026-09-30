"""Small helpers for writing generated markdown reports.

Several scripts write into the same document. Each owns one named section, delimited by
HTML comments (invisible when the markdown is rendered):

    <!-- section:audit:start -->
    ...generated content...
    <!-- section:audit:end -->

``write_section`` replaces that block if it exists and appends it otherwise, leaving the
rest of the file untouched.
"""

import re
from pathlib import Path

import numpy as np


def write_section(path, name, markdown):
    path = Path(path)
    start, end = f"<!-- section:{name}:start -->", f"<!-- section:{name}:end -->"
    block = f"{start}\n{markdown.strip()}\n{end}"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(lambda _: block, text)
    else:
        text = (text.rstrip() + "\n\n" if text.strip() else "") + block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def df_to_markdown(df, float_fmt="{:.2f}", index=True):
    """Render a DataFrame as a GitHub markdown table (no `tabulate` dependency)."""
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, *row in df.itertuples():
        cells = [str(idx)] if index else []
        for v in row:
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                cells.append("")
            elif isinstance(v, (float, np.floating)):
                cells.append(float_fmt.format(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
