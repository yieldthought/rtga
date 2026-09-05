"""Analyze matched oracle-navigation runs and write a two-page mini-paper.

Example: python -m rtga.oracle_report --input results/001-oracle/results.json
         --output papers/001-persistent-planning --stage confirmation
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shlex
import tempfile

import numpy as np


METHODS = {"persistent": "Persistent RTGA", "fresh": "Fresh evolution",
           "random": "Random shooting", "cem": "Categorical CEM"}
COLORS = {"persistent": "#8a1a1a", "fresh": "#467b91", "random": "#8a7750", "cem": "#72715e"}
METRICS = ("success", "restricted_hit", "mean_distance", "final_distance")


def analyze(data: dict, bootstrap_samples: int = 20000, bootstrap_seed: int = 20260905,
            allow_incomplete: bool = False) -> dict:
    """Return seed-level summaries and paired percentile-bootstrap intervals.

    Resampling units are matched environment seeds, never individual decisions.
    Restricted hit time is min(T, S): a run without a hit contributes S steps.
    """
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    protocol, records = data["protocol"], data["records"]
    variants, methods, seeds = protocol["variants"], protocol["methods"], protocol["seeds"]
    if "persistent" not in methods or not records:
        raise ValueError("Nonempty records and a persistent baseline are required")
    expected = {(v, s, m) for v in variants for s in seeds for m in methods}
    indexed = {}
    groups = defaultdict(list)
    for r in records:
        key = (r["variant"], r["seed"], r["method"])
        if key in indexed:
            raise ValueError(f"Duplicate run: {key}")
        if key not in expected:
            raise ValueError(f"Run not declared by protocol: {key}")
        if r["steps"] != protocol["steps"] or len(r["distance_curve"]) != r["steps"]:
            raise ValueError(f"Mismatched recording length: {key}")
        if len(r["latency_ms"]) != r["steps"]:
            raise ValueError(f"Missing per-decision latencies: {key}")
        hit = r["first_success_step"]
        if bool(r["success"]) != (hit is not None) or (hit is not None and not 1 <= hit <= r["steps"]):
            raise ValueError(f"Inconsistent success event: {key}")
        numeric = [r["mean_distance"], r["final_distance"], *r["distance_curve"], *r["latency_ms"]]
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"Nonfinite measurements: {key}")
        if not np.isclose(r["mean_distance"], np.mean(r["distance_curve"])):
            raise ValueError(f"Mean distance does not match recorded curve: {key}")
        if not np.isclose(r["final_distance"], r["distance_curve"][-1]):
            raise ValueError(f"Final distance does not match recorded curve: {key}")
        actual_hits = np.flatnonzero(np.asarray(r["distance_curve"]) <= protocol["success_radius"])
        actual_first = int(actual_hits[0]) + 1 if len(actual_hits) else None
        if actual_first != hit:
            raise ValueError(f"First success does not match recorded curve: {key}")
        indexed[key] = r
        groups[(r["variant"], r["method"])].append(r)
    missing = sorted(expected - indexed.keys())
    if missing and not allow_incomplete:
        raise ValueError(f"Incomplete suite: {len(indexed)}/{len(expected)} runs; first missing {missing[0]}")

    summaries = []
    for variant in variants:
        for method in methods:
            rows = sorted(groups[(variant, method)], key=lambda r: r["seed"])
            if not rows:
                continue
            hits = [r["first_success_step"] for r in rows if r["success"]]
            latencies = np.concatenate([r["latency_ms"] for r in rows])
            summaries.append({"variant": variant, "method": method, "n": len(rows),
                "successes": len(hits), "success": len(hits) / len(rows),
                "median_first_success": float(np.median(hits)) if hits else None,
                "restricted_hit": float(np.mean([r["first_success_step"] or r["steps"] for r in rows])),
                "mean_distance": float(np.mean([r["mean_distance"] for r in rows])),
                "final_distance": float(np.mean([r["final_distance"] for r in rows])),
                "model_transitions": sum(r["model_transitions"] for r in rows),
                "transitions_per_decision": float(np.mean([r["model_transitions"] / r["steps"] for r in rows])),
                "latency_p50_ms": float(np.percentile(latencies, 50)),
                "latency_p95_ms": float(np.percentile(latencies, 95)),
                "curve": np.mean([r["distance_curve"] for r in rows], axis=0).tolist()})

    comparisons = []
    rng = np.random.default_rng(bootstrap_seed)
    for variant in variants:
        for method in methods:
            if method == "persistent":
                continue
            matched = [s for s in seeds if (variant, s, "persistent") in indexed and (variant, s, method) in indexed]
            if not matched:
                continue
            a = [indexed[(variant, s, "persistent")] for s in matched]
            b = [indexed[(variant, s, method)] for s in matched]
            for ra, rb in zip(a, b):
                if not np.array_equal(ra["start"], rb["start"]) or not np.array_equal(ra["goal"], rb["goal"]):
                    raise ValueError(f"Initial conditions are not paired: {variant}, {ra['seed']}")
                ca = {k: v for k, v in ra["planner"].items() if k not in {"method", "seed"}}
                cb = {k: v for k, v in rb["planner"].items() if k not in {"method", "seed"}}
                if ca != cb:
                    raise ValueError(f"Planner budgets/settings differ: {variant}, {ra['seed']}, {method}")
            indices = rng.integers(len(matched), size=(bootstrap_samples, len(matched)))
            row = {"variant": variant, "baseline": method, "n": len(matched), "seeds": matched,
                   "equal_model_transitions": all(ra["model_transitions"] == rb["model_transitions"] for ra, rb in zip(a, b))}
            for metric in METRICS:
                def value(r):
                    return (r["first_success_step"] or r["steps"]) if metric == "restricted_hit" else float(r[metric])
                differences = np.array([value(ra) - value(rb) for ra, rb in zip(a, b)])
                boot = differences[indices].mean(axis=1)
                low, high = np.percentile(boot, [2.5, 97.5])
                row[metric] = {"mean": float(differences.mean()), "low": float(low), "high": float(high)}
            comparisons.append(row)
    return {"summaries": summaries, "comparisons": comparisons, "complete": not missing,
            "observed_runs": len(indexed), "expected_runs": len(expected), "missing": missing,
            "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed}


def _number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _interval(value, metric, markup=False):
    factor, digits = (100, 1) if metric == "success" else (1, 1) if metric == "restricted_hit" else (1, 3)
    point = f"{factor * value['mean']:+.{digits}f}"
    bounds = f"[{factor * value['low']:+.{digits}f}, {factor * value['high']:+.{digits}f}]"
    return f"{point}<small>{bounds}</small>" if markup else f"{point} {bounds}"


def _figure(analysis, variants) -> str:
    # Static scientific plot. No browser engine, remote assets or UI automation.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rtga-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "serif", "font.size": 9, "svg.fonttype": "none", "svg.hashsalt": "rtga-oracle-001"}):
        fig, axes = plt.subplots(1, len(variants), figsize=(7.1, 2.2), squeeze=False, sharey=True)
        styles = {"persistent": "-", "fresh": "--", "random": ":", "cem": "-."}
        for ax, variant in zip(axes[0], variants):
            for row in analysis["summaries"]:
                if row["variant"] != variant:
                    continue
                method = row["method"]
                ax.plot(np.arange(1, len(row["curve"]) + 1), row["curve"], label=METHODS.get(method, method),
                        color=COLORS.get(method, "#555555"), linestyle=styles.get(method, "-"),
                        linewidth=1.9 if method == "persistent" else 1.2)
            ax.set_title(variant, loc="left", fontstyle="italic", fontsize=11)
            ax.set_xlabel("Executed actions")
            ax.set_ylim(bottom=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#aaaaaa")
            ax.tick_params(length=3, color="#999999")
        axes[0, 0].set_ylabel("Mean distance to goal")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.015), ncols=min(4, len(labels)), frameon=False, fontsize=8)
        fig.subplots_adjust(left=.09, right=.99, top=.86, bottom=.29, wspace=.18)
        stream = io.StringIO()
        fig.savefig(stream, format="svg", transparent=True, metadata={"Date": None})
        plt.close(fig)
        return "<svg" + stream.getvalue().split("<svg", 1)[1]


def _reconstructed_command(protocol, output):
    c, seeds = protocol["planner"], protocol["seeds"]
    if seeds != list(range(min(seeds), min(seeds) + len(seeds))):
        return "Not reconstructible as one CLI invocation: nonconsecutive seeds; see recorded protocol."
    args = ["python", "-m", "rtga.cli", "oracle", "--output", str(output), "--seeds", str(len(seeds)),
            "--seed-start", str(min(seeds)), "--steps", str(protocol["steps"]),
            "--population", str(c["population"]), "--horizon", str(c["horizon"]),
            "--generations", str(c["generations"]), "--methods", *protocol["methods"],
            "--variants", *protocol["variants"], "--repeat", str(c.get("initial_repeat", 1))]
    return shlex.join(args)


def write_report(input_path, output, *, stage="development", run_command=None,
                 bootstrap_samples=20000, bootstrap_seed=20260905, allow_incomplete=False):
    """Write matching Markdown/HTML from one atomic result snapshot; return paths."""
    source, output = Path(input_path), Path(output)
    raw = source.read_bytes()
    data = json.loads(raw)
    analysis = analyze(data, bootstrap_samples, bootstrap_seed, allow_incomplete)
    p, provenance = data["protocol"], data.get("provenance", {})
    c = p["planner"]
    dirty = provenance.get("git_dirty")
    status = "Development data · not confirmatory" if stage == "development" else "Confirmation run"
    if not analysis["complete"]:
        status = f"INCOMPLETE DRAFT · {analysis['observed_runs']}/{analysis['expected_runs']} runs"
    state = "dirty working tree" if dirty else "clean working tree" if dirty is False else "working-tree status not recorded"
    status += " · " + state
    commands = run_command or provenance.get("command") or provenance.get("argv") or data.get("command")
    exact_command = commands is not None
    command = shlex.join(map(str, commands)) if isinstance(commands, list) else str(commands) if commands else _reconstructed_command(p, source.parent)
    if isinstance(provenance.get("argv"), list) and commands is provenance.get("argv"):
        command = " ".join([str(provenance.get("executable", "python")), shlex.join(map(str, commands))])
    checksum = hashlib.sha256(raw).hexdigest()
    commit = provenance.get("git_commit") or "not recorded"
    source_hashes = provenance.get("source_sha256") or provenance.get("source_hashes")
    clean_note = ("The run began from a clean recorded commit." if dirty is False else
                  "The run began with uncommitted changes; the commit alone does not identify the executed code." if dirty else
                  "Working-tree cleanliness was not recorded; the commit alone may not identify the executed code.")
    if source_hashes:
        clean_note += " The result file also contains per-source SHA-256 hashes."
    summaries, comparisons = analysis["summaries"], analysis["comparisons"]
    prose = []
    for variant in p["variants"]:
        rows = [r for r in summaries if r["variant"] == variant]
        persistent = next((r for r in rows if r["method"] == "persistent"), None)
        if persistent:
            outcomes = "; ".join(f"{METHODS.get(r['method'],r['method'])} {r['successes']}/{r['n']}" for r in rows)
            prose.append(f"In {variant}, observed successes were {outcomes}.")
    abstract = (f"We compared persistent action-plan populations with fresh evolution, random shooting and categorical CEM "
                f"using exact dynamics in a unit-square puck task. Each run contained {p['steps']} actions; "
                f"the protocol declared {len(p['seeds'])} paired seeds per environment. " + " ".join(prose) +
                " These tests isolate search behavior with a supplied objective. They do not test learned dynamics, curiosity or transfer.")
    weaknesses = []
    for r in summaries:
        persistent = next((x for x in summaries if x["variant"] == r["variant"] and x["method"] == "persistent"), None)
        if persistent and r["n"] == persistent["n"] and r["successes"] > persistent["successes"]:
            weaknesses.append(f"{METHODS.get(r['method'],r['method'])} reached more goals than persistence in {r['variant']} "
                              f"({r['successes']}/{r['n']} versus {persistent['successes']}/{persistent['n']})")
    tradeoff = ("Persistence did not dominate the observed success counts: " + "; ".join(weaknesses) + ". "
                "Improved distance control and reliable goal reachability are separate endpoints." if weaknesses else
                "Observed success counts alone do not resolve differences in approach speed or final stabilization; assess the distance and first-hit endpoints separately.")
    methods = (f"The agent selects among six primitive actions, simulates {c['population']} plans of {c['horizon']} steps "
               f"for {c['generations']} generations, executes the best first action, then replans. "
               f"Fitness is negative discounted mean future distance (discount {p.get('discount', .99)}); "
               f"success means observed distance ≤ {p.get('success_radius', .06)} at least once. "
               "Open has no internal wall; rooms has a permanent open doorway. Initial positions and goals are matched by seed.")
    population_method = ("Persistent RTGA shifts all genomes and appends a random action. Fresh evolution uses the same genetic "
                         "operators but initializes anew at each decision. Random shooting samples independent populations; "
                         "categorical CEM refits per-position action probabilities with a 10% uniform floor and keeps the "
                         "incumbent within each decision. CEM is initialized afresh between decisions, so this is not an iCEM comparison.")
    inference = (f"Every effect is the mean paired difference, persistent minus comparator, within one variant. "
                 f"The 95% percentile intervals resample complete seed pairs {bootstrap_samples:,} times "
                 f"(NumPy generator seed {bootstrap_seed}). Success effects are percentage points. "
                 f"Distance effects use world units. Restricted first-hit time is min(T, {p['steps']}); "
                 f"failures contribute {p['steps']} actions. Negative distance/time differences favor persistence. "
                 "Intervals are unadjusted across comparisons and endpoints; degenerate success intervals do not prove equal success probabilities.")
    limits = ("This is one simple environment family, one selected compute setting and known deterministic dynamics. "
              "The goal-directed distance objective provides dense guidance, and rooms does not require discovering a switch. "
              "Reaching the goal once is not the same as remaining there; mean and final distance expose that difference. "
              "The CEM baseline lacks cross-decision warm starts. Planning-time measurements exclude real environment steps, "
              "trace writing and training; they are not hard real-time guarantees. No learned-model or curiosity claim follows.")
    if stage == "development":
        limits += " These development seeds may inform configuration choices and must not be relabeled as held-out confirmation."
    totals = {int(r["transitions_per_decision"]) for r in summaries}
    compute = (f"Recorded per-decision model-transition counts: {', '.join(f'{x:,}' for x in sorted(totals))}. "
               f"The suite contains {sum(r['model_transitions'] for r in summaries):,} hypothetical transitions. "
               "Latency percentiles pool recorded decisions within each method/variant, rather than averaging run percentiles. "
               "Statistical intervals use seeds, not decisions, as independent units.")
    if not all(r["equal_model_transitions"] for r in comparisons):
        compute += " WARNING: paired model-transition budgets differ; interpret comparisons accordingly."

    table_head = ["Variant / method", "Success", "First hit†", "Mean dist.", "Final dist.", "Latency p50 / p95, ms"]
    rows = [[f"{r['variant']} / {METHODS.get(r['method'],r['method'])}", f"{r['successes']}/{r['n']}",
             _number(r["median_first_success"], 1), _number(r["mean_distance"]), _number(r["final_distance"]),
             f"{r['latency_p50_ms']:.2f} / {r['latency_p95_ms']:.2f}"] for r in summaries]
    pair_head = ["Variant / comparator", "Δ success, pp", "Δ restricted hit", "Δ mean distance", "Δ final distance"]
    pair_rows = [[f"{r['variant']} / {METHODS.get(r['baseline'],r['baseline'])}", *[_interval(r[m], m) for m in METRICS]] for r in comparisons]
    seeds_text = ", ".join(map(str, p["seeds"]))
    seed_rows = []
    for variant in p["variants"]:
        for method in p["methods"]:
            group = [r for r in data["records"] if r["variant"] == variant and r["method"] == method]
            success_ids = ", ".join(str(r["seed"]) for r in sorted(group,key=lambda r:r["seed"]) if r["success"])
            failed_ids = ", ".join(str(r["seed"]) for r in sorted(group,key=lambda r:r["seed"]) if not r["success"])
            seed_rows.append([variant, METHODS.get(method, method), success_ids or "none", failed_ids or "none"])
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    source_link = os.path.relpath(source.resolve(), output.parent.resolve())
    report_args = ["python", "-m", "rtga.oracle_report", "--input", str(source), "--output", str(output),
                   "--stage", stage, "--bootstrap-samples", str(bootstrap_samples),
                   "--bootstrap-seed", str(bootstrap_seed)]
    if run_command:
        report_args += ["--run-command", run_command]
    if allow_incomplete:
        report_args += ["--allow-incomplete"]
    report_command = shlex.join(report_args)
    svg = _figure(analysis, p["variants"])
    def md_table(head, body):
        return "| " + " | ".join(head) + " |\n| " + " | ".join(["---"] * len(head)) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in body)
    markdown = f"""# 001 · Persistent plans under an exact simulator

{status} · generated {generated}

## Abstract

{abstract}

## Methods

{methods}

{population_method}

## Results

{tradeoff}

{md_table(table_head, rows)}

† Median first-success step among successful runs only; not used for paired inference. Failed-run first-hit times remain censored at the interaction limit. Mean distance averages all executed steps, then equally weights seeds; final distance equally weights seeds. Success counts retain every completed seed.

{compute}

## Paired effects

{inference}

{md_table(pair_head, pair_rows)}

## Limitations

{limits}

## Reproduction and provenance

- Results: [{source_link}]({source_link})
- Result SHA-256: `{checksum}`
- Recorded run commit: `{commit}` ({state}). {clean_note}
- Run timestamp: `{provenance.get('created_utc', 'not recorded')}`
- Seeds: {seeds_text}
- Runtime: Python {provenance.get('python','?')}; NumPy {provenance.get('numpy','?')}; {provenance.get('platform','?')}
- Command below: {'recorded invocation' if exact_command else 'equivalent reconstruction from protocol, not a recorded historical invocation'}.

```sh
{command}
```

Report generation:

```sh
{report_command}
```

### Recorded source hashes

```json
{json.dumps(source_hashes, indent=2) if source_hashes else 'null'}
```

### Exact completed seed outcomes

{md_table(['Variant', 'Method', 'Successful seed IDs', 'Failed seed IDs'], seed_rows)}

## Related methods

[RHEA with a shift buffer, Gaina et al. 2017](https://rdgain.github.io/assets/pdf/papers/gaina2017rhhybrids.pdf) is the closest published planning mechanism. [iCEM, Pinneri et al.](https://proceedings.mlr.press/v155/pinneri21a.html) is a stronger continuous-control reference with elite memory and temporal sampling structure, not implemented in this comparison.
"""
    esc = html.escape
    def html_table(head, body, raw=False):
        return "<table><thead><tr>" + "".join(f"<th>{esc(x)}</th>" for x in head) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{x if raw and i else esc(x)}</td>" for i, x in enumerate(row)) + "</tr>" for row in body) + "</tbody></table>"
    html_pairs = [[f"{r['variant']} / {METHODS.get(r['baseline'],r['baseline'])}", *[_interval(r[m], m, True) for m in METRICS]] for r in comparisons]
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>001 · Persistent plans under an exact simulator</title><style>{_CSS}</style></head><body>
<article class="page"><header><div class="kicker">RTGA · experiment 001</div><h1>Persistent plans<br>under an exact simulator</h1><div class="meta">{esc(status)}<br>{esc(generated)} · {analysis['observed_runs']} recorded runs</div></header>
<div class="with-note"><section><h2>Abstract</h2><p>{esc(abstract)}</p></section><aside>Question: does keeping a living population improve navigation when the simulator and objective are already known?<br><br>This is the planner diagnostic. Learned models and autonomous curiosity are separate experiments.</aside></div>
<figure>{svg}<figcaption>Figure 1. Distance after each executed action, averaged equally across recorded seeds. Lines are descriptive means; paired intervals are on the next page.</figcaption></figure>
<h2>Recorded outcomes and planning cost</h2>{html_table(table_head, rows)}<p>{esc(tradeoff)}</p>
<p class="note">† First hit is the median among successful runs only. Mean/final distance include all runs. p50/p95 pool decision latencies within the row.</p>
<div class="with-note"><p>{esc(compute)}</p><aside>All methods execute one action per decision. Re-evaluating inherited plans is charged to the same model-transition budget.</aside></div></article>
<article class="page"><header><div class="kicker">RTGA · experiment 001 · inference and reproduction</div></header>
<h2>Paired differences: persistent minus comparator</h2>{html_table(pair_head, html_pairs, True)}
<div class="with-note"><p>{esc(inference)}</p><aside>Read the endpoints separately. More successes favor persistence; smaller distance and first-hit time favor persistence.<br><br>Each displayed interval contains 95% of bootstrap estimates. No multiplicity correction.</aside></div>
<h2>Methods</h2><p>{esc(methods)}</p><p>{esc(population_method)}</p>
<div class="with-note"><section><h2>Limits of this result</h2><p>{esc(limits)}</p></section><aside>Related work: <a href="https://rdgain.github.io/assets/pdf/papers/gaina2017rhhybrids.pdf">RHEA shift buffers (2017)</a> and <a href="https://proceedings.mlr.press/v155/pinneri21a.html">iCEM (2020)</a>. Categorical CEM here is an intentionally simple comparator.</aside></div>
<h2>Provenance</h2><p class="provenance">Run commit <code>{esc(commit)}</code> · {esc(state)}. {esc(clean_note)}<br>Results <a href="{esc(source_link,quote=True)}">{esc(source_link)}</a><br>Result SHA-256 <code>{checksum}</code><br>Seeds {esc(seeds_text)} · Python {esc(str(provenance.get('python','?')))} · NumPy {esc(str(provenance.get('numpy','?')))}<br>{esc(str(provenance.get('platform','?')))}<br>{'Recorded invocation' if exact_command else 'Equivalent command reconstructed from protocol'}:</p><pre>{esc(command)}</pre>
<p class="note">The Markdown companion records the report-generation command. Numerical results are derived from one {'complete' if analysis['complete'] else 'incomplete'} JSON snapshot; this paper does not select a favorable animation.</p></article></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    md_path, html_path = output.with_suffix(".md"), output.with_suffix(".html")
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(document, encoding="utf-8")
    return md_path, html_path


_CSS = """
@page { size:A4 portrait; margin:12mm 14mm 10mm; }
* { box-sizing:border-box; }
html,body { margin:0; padding:0; }
body { color:#181818; background:#fffff8; font:9.2pt/1.3 'Iowan Old Style','Palatino Linotype',Palatino,'Book Antiqua',Cambria,Georgia,serif; print-color-adjust:exact; -webkit-print-color-adjust:exact; }
.page { max-width:182mm; margin:0 auto; padding:0; break-after:page; }
.page:last-child { break-after:auto; }
header { margin-bottom:4mm; }
.kicker { font-size:8pt; letter-spacing:.08em; text-transform:uppercase; color:#666; }
h1 { font-size:24pt; line-height:1.05; font-weight:normal; margin:3mm 0; }
h2 { font-size:11.5pt; font-style:italic; font-weight:normal; margin:3.5mm 0 1.5mm; }
p { margin:1.5mm 0 2.5mm; }
.meta { border-top:.5pt solid #888; padding-top:2mm; color:#6f2929; font-size:8pt; }
.with-note { display:grid; grid-template-columns:3.1fr 1fr; gap:6mm; align-items:start; }
aside { font-size:8pt; line-height:1.35; color:#555; padding-top:3mm; }
figure { margin:3mm 0 2mm; }
svg { width:100%; height:auto; display:block; }
figcaption,.note { font-size:8pt; color:#555; }
table { width:100%; border-collapse:collapse; font-size:8.1pt; margin:2mm 0; font-variant-numeric:tabular-nums; }
th { border-block:.5pt solid #888; font-style:italic; font-weight:normal; text-align:right; padding:1.2mm .8mm; }
td { border-bottom:.3pt solid #d0d0ca; padding:1.1mm .8mm; text-align:right; white-space:nowrap; }
td:first-child,th:first-child { text-align:left; white-space:normal; }
small { display:block; font-size:7.8pt; color:#555; }
a { color:#6e2b2b; text-decoration:none; }
.provenance { font-size:8pt; overflow-wrap:anywhere; }
code,pre { font:7.5pt/1.3 ui-monospace,SFMono-Regular,Consolas,monospace; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; border-left:1pt solid #b7a19e; padding-left:2mm; margin:2mm 0; }
@media screen { body { padding:8mm 4mm; } .page+.page { border-top:1px solid #c8c8c0; margin-top:8mm; padding-top:6mm; } }
@media screen and (max-width:620px) { .with-note { grid-template-columns:1fr; gap:0; } aside { border-left:1px solid #ccc; padding:0 0 0 3mm; } table { font-size:7.8pt; } }
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", default=Path("papers/001-persistent-planning"), type=Path)
    parser.add_argument("--stage", choices=["development", "confirmation"], default="development")
    parser.add_argument("--run-command", help="Actual historical invocation, if not recorded in results")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    parser.add_argument("--allow-incomplete", action="store_true", help="Generate an explicitly incomplete draft")
    args = parser.parse_args()
    paths = write_report(args.input, args.output, stage=args.stage, run_command=args.run_command,
                         bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed,
                         allow_incomplete=args.allow_incomplete)
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
