#!/usr/bin/env python3
"""Render REPORT.md (plus a dot plot of the design-world runs) as a self-contained HTML page: report.html.
  .venv/bin/python tools/report_html.py"""
import collections
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from design_rescore import rescore  # noqa: E402
from design_summary import LABEL, tag  # noqa: E402

RIGHTS = {"oracle+authority", "self_org+knob-holders", "self_org+shared-knob-holders"}
ORDER = ["single", "isolated", "flat", "self_org", "oracle", "self_org+ownership", "self_org+ownership+reasoning",
         "oracle+authority", "self_org+knob-holders", "self_org+shared-knob-holders"]


def inline(t: str) -> str:
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", t)
    return t


def md_to_html(md: str) -> str:
    out, para, lst, table = [], [], [], []
    list_kind = None

    def flush():
        nonlocal para, lst, table, list_kind
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
        if lst:
            items = "".join(f"<li>{inline(x)}</li>" for x in lst)
            out.append(f"<{list_kind}>{items}</{list_kind}>")
        if table:
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in table if not re.fullmatch(r"\|?[-| :]+\|?", r.strip())]
            head, body = rows[0], rows[1:]
            cls = ' class="wide"' if len(head) > 8 else ""
            t = f"<div class=\"scroll\"><table{cls}><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
            for r in body:
                t += "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
            out.append(t + "</tbody></table></div>")
        para, lst, table, list_kind = [], [], [], None

    for line in md.split("\n"):
        s = line.rstrip()
        if s.startswith("<!--"):
            flush()
            if "DESIGNCHART" in s:
                out.append("%%CHART%%")
            continue
        if s.startswith("|"):
            if para or lst:
                flush()
            table.append(s)
            continue
        if table and not s.startswith("|"):
            flush()
        m = re.match(r"^(#{1,4}) (.*)", s)
        if m:
            flush()
            n = len(m.group(1))
            slug = re.sub(r"[^a-z0-9]+", "-", m.group(2).lower()).strip("-")
            out.append(f"<h{n} id=\"{slug}\">{inline(m.group(2))}</h{n}>")
            continue
        m = re.match(r"^(\s*)(?:[-*]|(\d+)\.) (.*)", s)
        if m and not m.group(1):
            kind = "ol" if m.group(2) else "ul"
            if para or (lst and list_kind != kind):
                flush()
            list_kind = kind
            lst.append(m.group(3))
            continue
        if lst and s.startswith("  ") and s.strip():
            lst[-1] += " " + s.strip()
            continue
        if not s.strip():
            flush()
            continue
        if lst:
            flush()
        para.append(s.strip())
    flush()
    return "\n".join(out)


def chart() -> str:
    by = collections.defaultdict(list)
    for d in sorted((ROOT / "runs_design").iterdir()):
        if (d / "run.json").exists():
            data = json.loads((d / "run.json").read_text())
            if len(data["snapshots"]) - 1 != 20:
                continue
            by[tag(data)].append(rescore(d))
    rows = [t for t in ORDER if t in by]
    lo, hi = -0.6, 0.8
    W, left, right, rh, top = 760, 330, 24, 34, 34
    H = top + rh * len(rows) + 30
    x = lambda v: left + (max(lo, min(hi, v)) - lo) / (hi - lo) * (W - left - right)
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="chart-title chart-desc" class="dots">',
           '<title id="chart-title">Design world: mean score over the second half of each run</title>',
           '<desc id="chart-desc">One dot per seed, a vertical bar at the condition mean. Filled dots: conditions with '
           'exclusive decision rights over knobs. Hollow dots: no rights.</desc>']
    for v in [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]:
        cls = "zero" if v == 0 else "grid"
        svg.append(f'<line class="{cls}" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top - 8}" y2="{H - 26}"/>')
        svg.append(f'<text class="tick" x="{x(v):.1f}" y="{H - 8}" text-anchor="middle">{v:+.1f}</text>'.replace("+0.0", "0"))
    svg.append(f'<text class="axis" x="{left}" y="16">second-half mean score (1 = optimum, 0 = random start)</text>')
    for k, t in enumerate(rows):
        y = top + rh * k + rh / 2
        rs = by[t]
        vals = [r["mean_second_half"] for r in rs]
        m = sum(vals) / len(vals)
        right_cls = "on" if t in RIGHTS else "off"
        if k % 2 == 0:
            svg.append(f'<rect class="band" x="0" y="{y - rh / 2:.1f}" width="{W}" height="{rh}"/>')
        svg.append(f'<text class="row {right_cls}" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">'
                   f'{html.escape(LABEL.get(t, t))[:54]}</text>')
        svg.append(f'<line class="mean {right_cls}" x1="{x(m):.1f}" x2="{x(m):.1f}" y1="{y - 10:.1f}" y2="{y + 10:.1f}">'
                   f'<title>{html.escape(LABEL.get(t, t))}: mean {m:+.2f} over {len(vals)} runs</title></line>')
        for r in rs:
            v = r["mean_second_half"]
            svg.append(f'<circle class="dot {right_cls}" cx="{x(v):.1f}" cy="{y:.1f}" r="5"><title>seed {r["seed"]}: '
                       f'{v:+.2f} (final {r["final"]:+.2f}, {r["reversals"]} shared-knob reversals)</title></circle>')
    svg.append("</svg>")
    legend = ('<div class="legend"><span><svg width="14" height="14" aria-hidden="true"><circle class="dot on" cx="7" cy="7" r="5"/></svg>'
              'exclusive decision rights over knobs</span><span><svg width="14" height="14" aria-hidden="true">'
              '<circle class="dot off" cx="7" cy="7" r="5"/></svg>no decision rights</span><span><svg width="14" height="14" '
              'aria-hidden="true"><line class="mean off" x1="7" x2="7" y1="1" y2="13"/></svg>condition mean</span></div>')
    return f'<figure class="chart"><div class="scroll">{"".join(svg)}</div>{legend}<figcaption>Design world, 50 agents, 20 ticks. One dot per seed; hover a dot for its final score and reversals. The table below has the same runs.</figcaption></figure>'


STYLE = """
/* Layout: one reading column with wide tables and the chart allowed to scroll inside their own boxes. */
:root {
  --bg: #fbfcfd; --surface: #f1f4f7; --fg: #17202b; --muted: #556170; --rule: #d9dfe6;
  --accent: #1d6fd8; --neutral: #a1aab5; --band: #f4f6f9;
  --font-body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #12161c; --surface: #1a2028; --fg: #e5eaf0; --muted: #9aa6b4; --rule: #2c3540;
  --accent: #4a90ea; --neutral: #5d6773; --band: #171c23; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #12161c; --surface: #1a2028; --fg: #e5eaf0; --muted: #9aa6b4; --rule: #2c3540;
  --accent: #4a90ea; --neutral: #5d6773; --band: #171c23; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 16px; line-height: 1.6;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 40px 64px; }
main { max-width: 860px; margin: 0 auto; }
h1 { font-size: 2rem; line-height: 1.2; margin: 0 0 12px; text-wrap: balance; letter-spacing: -0.01em; }
h2 { font-size: 1.35rem; margin: 48px 0 12px; padding-top: 16px; border-top: 1px solid var(--rule); text-wrap: balance; }
h3 { font-size: 1.08rem; margin: 32px 0 8px; text-wrap: balance; }
p, li { max-width: 72ch; }
p { margin: 0 0 14px; }
ul, ol { margin: 0 0 16px; padding-left: 1.3em; display: grid; gap: 8px; }
strong { font-weight: 600; }
code { font-family: var(--font-mono); font-size: 0.86em; background: var(--surface); padding: 1px 5px; border-radius: 4px; }
a { color: var(--accent); }
.meta { color: var(--muted); font-size: 0.95rem; }
.scroll { overflow-x: auto; margin: 0 0 18px; }
table { border-collapse: collapse; font-size: 0.88rem; font-variant-numeric: tabular-nums; min-width: 100%; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: 0.8rem; letter-spacing: 0.02em; white-space: nowrap; }
table.wide { font-size: 0.76rem; } table.wide td { white-space: nowrap; }
details { margin: 12px 0 0; } summary { cursor: pointer; color: var(--accent); font-weight: 600; }
summary:focus-visible, a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.chart { margin: 20px 0 24px; } .chart figcaption { color: var(--muted); font-size: 0.85rem; }
.dots { width: 100%; min-width: 640px; height: auto; display: block; font-family: var(--font-body); }
.dots .grid { stroke: var(--rule); stroke-width: 1; } .dots .zero { stroke: var(--muted); stroke-width: 1.5; }
.dots .band { fill: var(--band); }
.dots .tick { fill: var(--muted); font-size: 12px; font-family: var(--font-mono); }
.dots .axis { fill: var(--muted); font-size: 12px; }
.dots .row { fill: var(--fg); font-size: 13px; } .dots .row.off { fill: var(--muted); }
.dots .dot.on { fill: var(--accent); stroke: var(--bg); stroke-width: 2; }
.dots .dot.off, .legend .dot.off { fill: var(--bg); stroke: var(--neutral); stroke-width: 2; }
.legend .dot.on { fill: var(--accent); }
.dots .mean { stroke-width: 3; stroke-linecap: round; } .dots .mean.on { stroke: var(--accent); } .dots .mean.off, .legend .mean.off { stroke: var(--muted); stroke-width: 3; }
.dots circle:hover { stroke: var(--fg); }
.legend { display: flex; flex-wrap: wrap; gap: 8px 20px; font-size: 0.85rem; color: var(--muted); margin: 4px 0 6px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
@media (max-width: 520px) { h1 { font-size: 1.6rem; } body { font-size: 15px; } }
"""


def main():
    md = (ROOT / "REPORT.md").read_text()
    title_line, rest = md.split("\n", 1)
    meta, _, body = rest.strip().partition("\n\n")
    body = body.replace("<!--/DESIGNTABLE-->", "")
    body = body.replace("Scores swing from tick to tick", "<!--DESIGNCHART-->\n\nScores swing from tick to tick", 1)
    appendix_at = body.index("## Appendix: every run")
    main_html = md_to_html(body[:appendix_at])
    appendix = md_to_html(body[appendix_at:].split("\n", 1)[1])
    page = f"""<title>AlphaTeam Benchmark Report</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{STYLE}</style>
<main>
<h1>{inline(title_line.lstrip('# '))}</h1>
<p class="meta">{inline(' '.join(meta.split()))}</p>
{main_html.replace('%%CHART%%', chart())}
<h2 id="appendix-every-run">Appendix: every run</h2>
<details><summary>Show all {appendix.count('<tr>') - 1} runs</summary>
{appendix}
</details>
</main>
"""
    (ROOT / "report.html").write_text(page)
    print(f"wrote report.html ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
