#!/usr/bin/env python3
"""Render a live, auto-refreshing HTML dashboard from a JSON state file."""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PALETTE = [
    "rose",
    "coral",
    "apricot",
    "gold",
    "lime",
    "mint",
    "aqua",
    "sky",
    "blue",
    "violet",
    "mauve",
]

DEFAULT_RUN_COLUMNS = [
    {"key": "iteration", "label": "iteration"},
    {"key": "status", "label": "status"},
    {"key": "elapsed_seconds", "label": "seconds", "digits": 0},
    {"key": "pcc", "label": "PCC", "digits": 5},
    {"key": "decoder_device_ms_per_token_layer", "label": "device ms", "digits": 3},
    {"key": "artifact_dir", "label": "artifacts"},
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def fmt_value(value: Any, digits: int | None = None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if digits is not None:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return esc(value)
    return esc(value)


def inline_md(text: Any) -> str:
    """Tiny inline markdown renderer for code spans and links."""
    rest = "" if text is None else str(text)
    pieces: list[str] = []
    while rest:
        link_start = rest.find("[")
        code_start = rest.find("`")
        starts = [i for i in [link_start, code_start] if i >= 0]
        if not starts:
            pieces.append(esc(rest))
            break
        start = min(starts)
        pieces.append(esc(rest[:start]))
        if rest[start] == "`":
            after = rest[start + 1 :]
            code, sep, tail = after.partition("`")
            if sep:
                pieces.append(f"<code>{esc(code)}</code>")
                rest = tail
            else:
                pieces.append(esc(rest[start:]))
                break
        else:
            close = rest.find("]", start)
            open_paren = rest.find("(", close)
            close_paren = rest.find(")", open_paren)
            if close > start and open_paren == close + 1 and close_paren > open_paren:
                label = rest[start + 1 : close]
                href = rest[open_paren + 1 : close_paren]
                pieces.append(f'<a href="{esc(href)}">{esc(label)}</a>')
                rest = rest[close_paren + 1 :]
            else:
                pieces.append(esc(rest[start]))
                rest = rest[start + 1 :]
    return "".join(pieces)


def render_markdown(markdown: Any, empty: str = "") -> str:
    text = "" if markdown is None else str(markdown).strip()
    if not text:
        return f'<p class="muted">{esc(empty)}</p>' if empty else ""

    blocks: list[str] = []
    list_items: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(f"<p>{inline_md(' '.join(paragraph))}</p>")
            paragraph.clear()

    def flush_list() -> None:
        if list_items:
            blocks.append("<ul>" + "".join(list_items) + "</ul>")
            list_items.clear()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            flush_list()
        elif stripped.startswith("- "):
            flush_paragraph()
            list_items.append(f"<li>{inline_md(stripped[2:])}</li>")
        else:
            flush_list()
            paragraph.append(stripped)
    flush_paragraph()
    flush_list()
    return "".join(blocks)


def extract_agent_knowledge(skill_path: Any) -> str:
    if not skill_path:
        return ""
    path = Path(str(skill_path)).expanduser()
    if not path.exists():
        return ""
    text = path.read_text()
    marker = "## Agent Knowledge"
    if marker not in text:
        return ""
    section = text.split(marker, 1)[1]
    next_heading = section.find("\n## ")
    if next_heading >= 0:
        section = section[:next_heading]
    return section.strip()


def knowledge_entries(markdown: Any) -> list[tuple[str, list[str]]]:
    """Parse strict bullet/vote Agent Knowledge, with a small heading fallback for old notes."""
    lines = ("" if markdown is None else str(markdown)).splitlines()
    entries: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_body: list[str] = []

    for line in lines:
        stripped = line.strip()
        if line.startswith("- ") and not stripped.startswith("- UP ") and not stripped.startswith("- DOWN "):
            if current_title is not None:
                entries.append((current_title, current_body))
            current_title = line[2:].strip()
            current_body = []
        elif current_title is not None:
            current_body.append(stripped)
    if current_title is not None:
        entries.append((current_title, current_body))
    if entries:
        return entries

    current_title = None
    current_body = []
    for line in lines:
        if line.startswith("### "):
            if current_title is not None:
                entries.append((current_title, current_body))
            current_title = line[4:].strip()
            current_body = []
        elif current_title is not None:
            current_body.append(line.strip())
    if current_title is not None:
        entries.append((current_title, current_body))
    return entries


def render_knowledge(markdown: Any, empty: str = "No skill knowledge recorded yet.") -> str:
    entries = knowledge_entries(markdown)
    if not entries:
        return f'<p class="muted">{esc(empty)}</p>'

    cards: list[str] = []
    for index, (title, body_lines) in enumerate(entries):
        votes: list[str] = []
        body: list[str] = []
        for line in body_lines:
            stripped = line.strip()
            vote = stripped[2:] if stripped.startswith("- ") else stripped
            if vote.startswith("UP ") or vote.startswith("DOWN "):
                votes.append(vote)
            elif stripped and stripped != "This section is maintained by agents. Keep entries short, general, and evidence-based.":
                body.append(stripped)
        body_html = render_markdown("\n".join(body))
        vote_html = ""
        if votes:
            vote_html = '<ul class="votes">' + "".join(f"<li>{inline_md(v)}</li>" for v in votes) + "</ul>"
        cards.append(
            f'<article class="knowledge-entry k{index % len(PALETTE)}">'
            f"<h3>{inline_md(title)}</h3>{body_html}{vote_html}</article>"
        )
    return '<div class="knowledge-grid">' + "\n".join(cards) + "</div>"


def read_text_path(path_value: Any) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value)).expanduser()
    return path.read_text() if path.exists() else ""


def knowledge_from_diff(path_value: Any) -> str:
    path = Path(str(path_value)).expanduser() if path_value else None
    if not path or not path.exists():
        return ""
    collected: list[str] = []
    active = False
    for line in path.read_text().splitlines():
        if line.startswith("+++"):
            continue
        if line.startswith("+- ") or line.startswith("+### "):
            active = True
        if active:
            if line.startswith("+"):
                collected.append(line[1:])
            elif line.startswith("@@") or line.startswith("diff ") or line.startswith("---"):
                active = False
    return "\n".join(collected).strip()


def change_knowledge(change: dict[str, Any]) -> str:
    if change.get("knowledge"):
        return str(change["knowledge"])
    if change.get("knowledge_path"):
        return read_text_path(change["knowledge_path"])
    return knowledge_from_diff(change.get("diff_path"))


def normalize_column(column: Any) -> dict[str, Any]:
    if isinstance(column, str):
        return {"key": column, "label": column.replace("_", " ")}
    return dict(column)


def infer_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return [{"key": key, "label": key.replace("_", " ")} for key in keys]


def render_table(columns: list[Any], rows: list[dict[str, Any]], empty: str = "No rows yet.") -> str:
    if not rows:
        col_count = max(len(columns), 1)
        return f'<table><tbody><tr><td colspan="{col_count}" class="muted">{esc(empty)}</td></tr></tbody></table>'
    normalized = [normalize_column(c) for c in (columns or infer_columns(rows))]
    headers = "".join(f"<th>{esc(c.get('label') or c.get('key'))}</th>" for c in normalized)
    body_rows: list[str] = []
    for row_index, row in enumerate(rows):
        status_class = esc(row.get("status", ""))
        cells = []
        for column in normalized:
            key = column.get("key")
            value = row.get(key)
            digits = column.get("digits")
            if column.get("path") and value:
                value_html = f'<code>{esc(value)}</code>'
            elif column.get("markdown"):
                value_html = inline_md(value)
            else:
                value_html = fmt_value(value, digits)
            cells.append(f"<td>{value_html}</td>")
        body_rows.append(f'<tr class="row-{row_index % len(PALETTE)} {status_class}">' + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def source_rows(data: dict[str, Any], source: str) -> list[dict[str, Any]]:
    value = data.get(source, [])
    return value if isinstance(value, list) else []


def spark(rows: list[dict[str, Any]], spec: dict[str, Any]) -> str:
    y_key = spec.get("y_key")
    x_key = spec.get("x_key", "iteration")
    pts: list[tuple[Any, float]] = []
    for index, row in enumerate(rows):
        raw_y = row.get(y_key)
        if isinstance(raw_y, (int, float)):
            pts.append((row.get(x_key, index), float(raw_y)))
    if not pts:
        return '<p class="muted">awaiting data</p>'

    nums = [v for _, v in pts]
    lo, hi = min(nums), max(nums)
    span = hi - lo or 1.0
    x0, x1 = 50, 198
    y0, y1 = 22, 78
    count_span = max(len(pts) - 1, 1)

    def x_at(position: int) -> float:
        return x0 + (position / count_span) * (x1 - x0)

    def y_at(value: float) -> float:
        return y1 - ((value - lo) / span) * (y1 - y0)

    unit = spec.get("unit", "")
    tick_digits = int(spec.get("tick_digits", spec.get("digits", 2)))
    latest_digits = int(spec.get("latest_digits", spec.get("digits", 2)))
    ticks: list[str] = []
    for value in [hi, (hi + lo) / 2, lo]:
        y = y_at(value)
        ticks.append(
            f'<line class="grid-line" x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}"></line>'
            f'<line class="tick-line" x1="{x0 - 4}" x2="{x0}" y1="{y:.1f}" y2="{y:.1f}"></line>'
            f'<text class="axis-label" x="{x0 - 7}" y="{y + 2.5:.1f}" text-anchor="end">'
            f"{fmt_value(value, tick_digits)}{esc(unit)}</text>"
        )

    point_html: list[str] = []
    path_parts: list[str] = []
    for position, (x_label, value) in enumerate(pts):
        x = x_at(position)
        y = y_at(value)
        path_parts.append(("M" if position == 0 else "L") + f" {x:.1f} {y:.1f}")
        point_html.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2"></circle>')
        point_html.append(f'<text class="x-label" x="{x:.1f}" y="93" text-anchor="middle">{esc(x_label)}</text>')
    path = f'<path class="series" d="{" ".join(path_parts)}"></path>' if len(path_parts) > 1 else ""
    label = f"{fmt_value(nums[-1], latest_digits)}{unit} latest"
    return (
        '<svg class="spark" viewBox="0 0 220 100" role="img">'
        f'<text class="latest-label" x="{x0}" y="12">{esc(label)}</text>'
        f'{"".join(ticks)}'
        f'<line class="axis-line" x1="{x0}" x2="{x0}" y1="{y0}" y2="{y1}"></line>'
        f'<line class="axis-line" x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}"></line>'
        f"{path}{''.join(point_html)}</svg>"
    )


def bar_chart(rows: list[dict[str, Any]], spec: dict[str, Any]) -> str:
    y_key = spec.get("y_key")
    x_key = spec.get("x_key", "model")
    group_key = spec.get("group_key", "condition")
    points = [
        (str(row.get(x_key, "")), str(row.get(group_key, "")), float(row[y_key]))
        for row in rows
        if isinstance(row.get(y_key), (int, float))
    ]
    if not points:
        return '<p class="muted">awaiting data</p>'

    categories = list(dict.fromkeys(category for category, _, _ in points))
    groups = list(dict.fromkeys(group for _, group, _ in points))
    values = {(category, group): value for category, group, value in points}
    hi = max(value for _, _, value in points)
    hi = hi or 1.0
    x0, x1 = 50, 214
    y0, y1 = 20, 82
    category_width = (x1 - x0) / max(len(categories), 1)
    bar_width = min(13.0, (category_width * 0.72) / max(len(groups), 1))
    unit = spec.get("unit", "")
    tick_digits = int(spec.get("tick_digits", spec.get("digits", 2)))
    latest_digits = int(spec.get("latest_digits", spec.get("digits", 2)))

    def y_at(value: float) -> float:
        return y1 - (value / hi) * (y1 - y0)

    ticks: list[str] = []
    for value in [hi, hi / 2, 0]:
        y = y_at(value)
        ticks.append(
            f'<line class="grid-line" x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}"></line>'
            f'<line class="tick-line" x1="{x0 - 4}" x2="{x0}" y1="{y:.1f}" y2="{y:.1f}"></line>'
            f'<text class="axis-label" x="{x0 - 7}" y="{y + 2.5:.1f}" text-anchor="end">'
            f"{fmt_value(value, tick_digits)}{esc(unit)}</text>"
        )

    bars: list[str] = []
    for category_index, category in enumerate(categories):
        category_center = x0 + category_width * (category_index + 0.5)
        group_total_width = bar_width * len(groups)
        for group_index, group in enumerate(groups):
            value = values.get((category, group))
            if value is None:
                continue
            x = category_center - group_total_width / 2 + group_index * bar_width
            y = y_at(value)
            height = y1 - y
            bars.append(
                f'<rect class="bar bar-{group_index % len(PALETTE)}" x="{x:.1f}" y="{y:.1f}" '
                f'width="{bar_width - 1:.1f}" height="{height:.1f}">'
                f"<title>{esc(category)} {esc(group)}: {fmt_value(value, latest_digits)}{esc(unit)}</title>"
                "</rect>"
            )
        bars.append(f'<text class="x-label" x="{category_center:.1f}" y="97" text-anchor="middle">{esc(category)}</text>')

    legend_items: list[str] = []
    for index, group in enumerate(groups):
        lx = x0 + index * 58
        legend_items.append(
            f'<rect class="bar bar-{index % len(PALETTE)}" x="{lx}" y="6" width="7" height="7"></rect>'
            f'<text class="legend-label" x="{lx + 10}" y="12">{esc(group)}</text>'
        )

    return (
        '<svg class="spark bar-spark" viewBox="0 0 240 108" role="img">'
        f'{"".join(legend_items)}{"".join(ticks)}'
        f'<line class="axis-line" x1="{x0}" x2="{x0}" y1="{y0}" y2="{y1}"></line>'
        f'<line class="axis-line" x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}"></line>'
        f'{"".join(bars)}</svg>'
    )


def default_graphs(data: dict[str, Any]) -> list[dict[str, Any]]:
    runs = data.get("runs", [])
    if not isinstance(runs, list):
        return []
    keys = {key for row in runs if isinstance(row, dict) for key in row}
    graphs: list[dict[str, Any]] = []
    if "elapsed_seconds" in keys:
        graphs.append(
            {
                "title": "time to solution",
                "source": "runs",
                "x_key": "iteration",
                "y_key": "elapsed_seconds",
                "unit": "s",
                "digits": 0,
            }
        )
    if "decoder_device_ms_per_token_layer" in keys:
        graphs.append(
            {
                "title": "decoder performance",
                "source": "runs",
                "x_key": "iteration",
                "y_key": "decoder_device_ms_per_token_layer",
                "unit": " ms",
                "digits": 4,
            }
        )
    return graphs


def meta_items(data: dict[str, Any], output_path: Path) -> list[dict[str, Any]]:
    if isinstance(data.get("meta"), list) and data["meta"]:
        return data["meta"][:6]
    remote = data.get("remote", {}) if isinstance(data.get("remote"), dict) else {}
    items = [
        {"label": "status", "value": data.get("status", "active")},
        {"label": "updated", "value": data.get("updated_utc", utc_now())},
    ]
    host = remote.get("machine") or remote.get("host")
    if host:
        port = f":{remote['ssh_port']}" if remote.get("ssh_port") else ""
        items.append({"label": "host", "value": f"{host}{port}"})
    if remote.get("tt_metal_path"):
        items.append({"label": "tt-metal", "value": remote.get("tt_metal_path"), "code": True})
    items.append({"label": "html", "value": str(output_path), "code": True})
    return items[:6]


def render_dashboard(data: dict[str, Any], output_path: Path, refresh_seconds: int | None = None) -> str:
    refresh = int(refresh_seconds or data.get("refresh_seconds") or 60)
    data_updated = data.get("updated_utc") or utc_now()
    data["updated_utc"] = data_updated

    steps = data.get("steps", []) if isinstance(data.get("steps"), list) else []
    events = data.get("events", []) if isinstance(data.get("events"), list) else []
    runs = data.get("runs", []) if isinstance(data.get("runs"), list) else []
    graphs = data.get("graphs") if isinstance(data.get("graphs"), list) else default_graphs(data)
    tables = data.get("tables", []) if isinstance(data.get("tables"), list) else []
    sections = data.get("sections", []) if isinstance(data.get("sections"), list) else []
    skill_changes = data.get("skill_changes", []) if isinstance(data.get("skill_changes"), list) else []
    current_knowledge = data.get("current_skill_knowledge") or extract_agent_knowledge(data.get("skill_path"))
    acceptance = data.get("acceptance") or data.get("notes") or ""

    meta_html = "".join(
        "<div>"
        f'<span class="muted">{esc(item.get("label"))}</span><br>'
        + (f"<code>{esc(item.get('value'))}</code>" if item.get("code") else esc(item.get("value")))
        + "</div>"
        for item in meta_items(data, output_path)
    )

    step_rows = "".join(
        f'<tr class="{esc(step.get("status", ""))}"><td><span class="dot"></span>{esc(step.get("name"))}</td>'
        f"<td>{esc(step.get('status'))}</td></tr>"
        for step in steps
    ) or '<tr><td class="muted">No plan steps yet.</td><td></td></tr>'

    run_columns = data.get("run_columns") or [c for c in DEFAULT_RUN_COLUMNS if any(c["key"] in r for r in runs)]
    run_table = render_table(run_columns, runs, empty="No runs yet.") if runs or data.get("show_empty_runs", True) else ""

    graph_html = "".join(
        f'<div class="graph g{index % len(PALETTE)}"><h2>{esc(graph.get("title"))}</h2>'
        f'{bar_chart(source_rows(data, graph.get("source", "runs")), graph) if graph.get("type") == "bar" else spark(source_rows(data, graph.get("source", "runs")), graph)}</div>'
        for index, graph in enumerate(graphs)
    )
    if graph_html:
        graph_html = f'<div class="graphs">{graph_html}</div>'

    event_rows = "".join(
        f'<tr><td>{esc(event.get("time"))}</td><td>{esc(event.get("kind"))}</td>'
        f"<td>{inline_md(event.get('summary'))}</td></tr>"
        for event in events[-24:]
    ) or '<tr><td colspan="3" class="muted">No activity yet.</td></tr>'

    table_html = "".join(
        f'<h2>{esc(table.get("title"))}</h2>'
        f'{render_table(table.get("columns", []), table.get("rows", []), empty=table.get("empty", "No rows yet."))}'
        for table in tables
        if isinstance(table, dict)
    )

    section_html = "".join(
        f'<h2>{esc(section.get("title"))}</h2><section class="prose">{render_markdown(section.get("markdown"))}</section>'
        for section in sections
        if isinstance(section, dict)
    )

    skill_change_html = "".join(
        '<section class="skill-change">'
        f'<div class="change-meta"><span>iteration {esc(change.get("iteration", ""))}</span>'
        f'<span>{esc(change.get("decision", ""))}</span></div>'
        f'{render_knowledge(change_knowledge(change), empty=change.get("summary") or "No promoted knowledge text recorded.")}'
        "</section>"
        for change in skill_changes
        if isinstance(change, dict)
    ) or '<p class="muted">No skill changes recorded yet.</p>'

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{refresh}">
<title>{esc(data.get("title", "Live Dashboard"))}</title>
<style>
@page {{ size: A4 portrait; margin: 12mm; }}
:root {{
  --paper: #0b0f18;
  --paper-2: #111827;
  --paper-3: #192034;
  --ink: #fff7ed;
  --muted: #c9c2d5;
  --rule: rgba(255, 247, 237, 0.16);
  --rose: #ff9fb7;
  --coral: #ffb28f;
  --apricot: #ffd19a;
  --gold: #f5e58d;
  --lime: #c8ef9a;
  --mint: #99ecc0;
  --aqua: #89efe5;
  --sky: #96ceff;
  --blue: #9faeff;
  --violet: #c5a7ff;
  --mauve: #f0a7ff;
}}
html, body {{ margin: 0; padding: 0; background: var(--paper); color: var(--ink); }}
body {{
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Cambria, Georgia, serif;
  font-size: 14px;
  line-height: 1.36;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}}
main {{ max-width: 1160px; margin: 0 auto; padding: 28px 24px 42px; position: relative; }}
main::before {{
  content: "";
  display: block;
  height: 4px;
  margin-bottom: 18px;
  background: linear-gradient(90deg,
    var(--rose) 0 9%, var(--coral) 9% 18%, var(--apricot) 18% 27%,
    var(--gold) 27% 36%, var(--lime) 36% 45%, var(--mint) 45% 54%,
    var(--aqua) 54% 63%, var(--sky) 63% 72%, var(--blue) 72% 81%,
    var(--violet) 81% 90%, var(--mauve) 90% 100%);
}}
h1 {{ color: #fff8ef; font-weight: 400; font-size: 34px; line-height: 1.05; margin: 0 0 7px; }}
h2 {{ color: var(--gold); font-weight: 400; font-style: italic; font-size: 18px; margin: 26px 0 8px; }}
h3 {{ margin: 0 0 4px; font-size: 14px; font-style: italic; font-weight: 400; color: var(--ink); }}
.meta {{ border-top: 1px solid var(--rose); border-bottom: 1px solid var(--aqua); padding: 8px 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; font-size: 12px; background: linear-gradient(90deg, rgba(255,159,183,0.08), rgba(137,239,229,0.07), rgba(197,167,255,0.08)); }}
.layout {{ display: grid; grid-template-columns: 7fr 4fr; gap: 26px; align-items: start; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ border-bottom: 1px solid var(--rule); padding: 5px 6px; text-align: left; vertical-align: top; }}
th {{ color: #fff4c7; border-bottom-color: rgba(255,244,199,0.5); font-style: italic; font-weight: 400; }}
tbody tr:nth-child(11n+1) td:first-child {{ border-left: 3px solid var(--rose); }}
tbody tr:nth-child(11n+2) td:first-child {{ border-left: 3px solid var(--coral); }}
tbody tr:nth-child(11n+3) td:first-child {{ border-left: 3px solid var(--apricot); }}
tbody tr:nth-child(11n+4) td:first-child {{ border-left: 3px solid var(--gold); }}
tbody tr:nth-child(11n+5) td:first-child {{ border-left: 3px solid var(--lime); }}
tbody tr:nth-child(11n+6) td:first-child {{ border-left: 3px solid var(--mint); }}
tbody tr:nth-child(11n+7) td:first-child {{ border-left: 3px solid var(--aqua); }}
tbody tr:nth-child(11n+8) td:first-child {{ border-left: 3px solid var(--sky); }}
tbody tr:nth-child(11n+9) td:first-child {{ border-left: 3px solid var(--blue); }}
tbody tr:nth-child(11n+10) td:first-child {{ border-left: 3px solid var(--violet); }}
tbody tr:nth-child(11n) td:first-child {{ border-left: 3px solid var(--mauve); }}
.panel {{ border-left: 3px solid var(--sky); padding-left: 12px; background: linear-gradient(180deg, rgba(150,206,255,0.08), rgba(240,167,255,0.05)); }}
.active .dot {{ background: var(--gold); }}
.complete .dot, .success .dot {{ background: var(--mint); }}
.blocked .dot, .failed .dot, .failure .dot {{ background: var(--rose); }}
.pending .dot {{ background: var(--rule); }}
.dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 8px; margin-right: 8px; background: var(--rule); }}
.muted {{ color: var(--muted); font-style: italic; }}
.graphs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 10px; }}
.graph {{ background: rgba(255,247,237,0.025); padding: 7px 8px 4px; border-top: 2px solid var(--coral); }}
.g1 {{ border-top-color: var(--violet); }}
.g2 {{ border-top-color: var(--aqua); }}
.g3 {{ border-top-color: var(--lime); }}
.spark {{ width: 100%; height: 130px; }}
.spark .series {{ fill: none; stroke: var(--gold); stroke-width: 2; }}
.g1 .spark .series {{ stroke: var(--aqua); }}
.g2 .spark .series {{ stroke: var(--mauve); }}
.g3 .spark .series {{ stroke: var(--lime); }}
.spark .axis-line {{ stroke: rgba(255,247,237,0.58); stroke-width: 0.8; }}
.spark .tick-line {{ stroke: rgba(255,247,237,0.46); stroke-width: 0.8; }}
.spark .grid-line {{ stroke: rgba(255,247,237,0.11); stroke-width: 0.6; }}
.spark circle {{ fill: var(--mauve); stroke: var(--paper); stroke-width: 0.8; }}
.spark .bar {{ stroke: var(--paper); stroke-width: 0.6; }}
.spark .bar-0 {{ fill: var(--gold); }}
.spark .bar-1 {{ fill: var(--aqua); }}
.spark .bar-2 {{ fill: var(--mauve); }}
.spark .bar-3 {{ fill: var(--lime); }}
.spark text {{ font: 7px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; fill: var(--ink); }}
.spark .axis-label, .spark .x-label {{ fill: var(--muted); font-size: 6.5px; }}
.spark .latest-label {{ fill: var(--ink); font-size: 7.5px; }}
.spark .legend-label {{ fill: var(--ink); font-size: 6.5px; }}
.skill-change {{ border-top: 1px solid var(--rule); padding-top: 8px; margin-top: 8px; }}
.change-meta {{ display: flex; gap: 12px; color: var(--muted); font-style: italic; font-size: 12px; margin-bottom: 6px; }}
.knowledge-grid {{ display: grid; grid-template-columns: 1fr; gap: 8px; }}
.knowledge-entry {{ border-left: 3px solid var(--aqua); padding: 1px 0 2px 10px; background: rgba(255,247,237,0.018); }}
.knowledge-entry p {{ margin: 3px 0; }}
.knowledge-entry .votes {{ margin: 4px 0 0 0; padding-left: 16px; color: var(--muted); font-size: 12px; }}
.k0 {{ border-left-color: var(--rose); }}
.k1 {{ border-left-color: var(--coral); }}
.k2 {{ border-left-color: var(--apricot); }}
.k3 {{ border-left-color: var(--gold); }}
.k4 {{ border-left-color: var(--lime); }}
.k5 {{ border-left-color: var(--mint); }}
.k6 {{ border-left-color: var(--aqua); }}
.k7 {{ border-left-color: var(--sky); }}
.k8 {{ border-left-color: var(--blue); }}
.k9 {{ border-left-color: var(--violet); }}
.k10 {{ border-left-color: var(--mauve); }}
.prose p {{ margin: 4px 0 8px; }}
a {{ color: var(--sky); }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.9em; }}
section, aside {{ background: transparent; }}
@media (max-width: 900px) {{
  main {{ padding: 20px 16px 34px; }}
  .layout {{ grid-template-columns: 1fr; }}
  h1 {{ font-size: 28px; }}
}}
</style>
<script>
setTimeout(function () {{ window.location.reload(); }}, {refresh} * 1000);
</script>
</head>
<body>
<main>
<h1>{esc(data.get("title", "Live Dashboard"))}</h1>
<div class="meta">{meta_html}</div>
<div class="layout">
  <section>
    <h2>{esc(data.get("runs_title", "runs"))}</h2>
    {run_table}
    {graph_html}
    {table_html}
    {section_html}
    <h2>activity</h2>
    <table><thead><tr><th>time</th><th>kind</th><th>summary</th></tr></thead><tbody>{event_rows}</tbody></table>
    <h2>skill changes</h2>
    {skill_change_html}
    <h2>current agent knowledge</h2>
    {render_knowledge(current_knowledge)}
  </section>
  <aside class="panel">
    <h2>plan progress</h2>
    <table><tbody>{step_rows}</tbody></table>
    <h2>notes</h2>
    <section class="prose">{render_markdown(acceptance, empty="No notes yet.")}</section>
  </aside>
</div>
</main>
</body>
</html>
"""
    return html_doc


def sample_state(title: str, refresh_seconds: int) -> dict[str, Any]:
    now = utc_now()
    return {
        "title": title,
        "status": "active",
        "updated_utc": now,
        "refresh_seconds": refresh_seconds,
        "meta": [
            {"label": "status", "value": "active"},
            {"label": "updated", "value": now},
            {"label": "owner", "value": "Codex"},
            {"label": "report", "value": "dashboard.html", "code": True},
        ],
        "steps": [
            {"name": "Create dashboard", "status": "complete"},
            {"name": "Run experiment", "status": "active"},
            {"name": "Review artifacts", "status": "pending"},
        ],
        "runs": [
            {
                "iteration": 0,
                "status": "active",
                "elapsed_seconds": None,
                "artifact_dir": "artifacts/run-0",
            }
        ],
        "graphs": [
            {
                "title": "time to solution",
                "source": "runs",
                "x_key": "iteration",
                "y_key": "elapsed_seconds",
                "unit": "s",
                "digits": 0,
            }
        ],
        "events": [
            {"time": now, "kind": "dashboard", "summary": "Initialized live dashboard state."}
        ],
        "skill_changes": [],
        "acceptance": "Replace this note with the success criteria and evidence requirements for the task.",
    }


def render_once(state_path: Path, output_path: Path, refresh_seconds: int | None) -> None:
    data = json.loads(state_path.read_text())
    html_doc = render_dashboard(data, output_path, refresh_seconds=refresh_seconds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc)
    print(output_path)


def init_state(state_path: Path, output_path: Path, title: str, refresh_seconds: int) -> None:
    if not state_path.exists():
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(sample_state(title, refresh_seconds), indent=2) + "\n")
    render_once(state_path, output_path, refresh_seconds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", nargs="?", help="dashboard_state.json to render")
    parser.add_argument("--init", help="create this dashboard_state.json if it does not exist")
    parser.add_argument("-o", "--output", help="HTML output path")
    parser.add_argument("--title", default="Live Dashboard", help="title for --init")
    parser.add_argument("--refresh-seconds", type=int, default=None, help="HTML auto-refresh interval")
    parser.add_argument("--watch", action="store_true", help="rerender repeatedly")
    parser.add_argument("--interval", type=float, default=10.0, help="seconds between watch renders")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state_arg = args.init or args.state
    if not state_arg:
        print("error: provide a state path or --init path", file=sys.stderr)
        return 2
    state_path = Path(state_arg).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else state_path.with_suffix(".html")
    refresh = args.refresh_seconds if args.refresh_seconds is not None else 60

    if args.init:
        init_state(state_path, output_path, args.title, refresh)
    elif not state_path.exists():
        print(f"error: state file does not exist: {state_path}", file=sys.stderr)
        return 2
    else:
        render_once(state_path, output_path, args.refresh_seconds)

    if args.watch:
        while True:
            time.sleep(args.interval)
            render_once(state_path, output_path, args.refresh_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
