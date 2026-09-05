"""Generate the two-page print companion to the recorded model study."""

from __future__ import annotations

import argparse
from html import escape
from io import StringIO
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rtga-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"open": "#37657b", "rooms": "#8b4b45"}


def _svg(figure) -> str:
    buffer = StringIO()
    figure.savefig(buffer, format="svg", transparent=True)
    plt.close(figure)
    svg = buffer.getvalue()
    return svg[svg.index("<svg"):]


def _learning_figure(runs: list[dict]) -> str:
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.10))
    for axis, measure, title in zip(axes, ("one_step", "rollout24"), ("One-step physical error", "24-step endpoint error")):
        for variant in ("open", "rooms"):
            selected = [run for run in runs if run["variant"] == variant]
            values = np.array([
                [point["one_step"]["physical_mse"] if measure == "one_step"
                 else point["rollouts"]["24"]["endpoint_physical_mse"] for point in run["checkpoints"]]
                for run in selected
            ])
            for row in values:
                axis.plot([0, 1, 2], row, color=COLORS[variant], alpha=.16, linewidth=.65)
            mean = values.mean(axis=0)
            axis.plot([0, 1, 2], mean, color=COLORS[variant], linewidth=1.4, marker="o", markersize=3)
            offset = .80 if variant == "open" and measure == "rollout24" else 1.20 if measure == "rollout24" else 1
            axis.text(2.12, mean[-1] * offset, variant, color=COLORS[variant], fontsize=8.5, va="center")
        axis.set_yscale("log")
        axis.set_xlim(-.12, 2.66)
        axis.set_xticks([0, 1, 2], ["0", "100", "500"])
        axis.set_xlabel("Fitting batches", fontsize=8.5)
        axis.set_title(title, loc="left", fontsize=10, pad=8)
        axis.set_ylabel("MSE · log scale", fontsize=8)
        axis.tick_params(axis="both", labelsize=8, width=.4, length=2.5)
        axis.minorticks_off()
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_linewidth(.4)
            axis.spines[side].set_color("#888")
    figure.subplots_adjust(left=.09, right=.985, top=.83, bottom=.24, wspace=.35)
    return _svg(figure)


def _uncertainty_figure(groups: list[tuple], runs: list[dict]) -> str:
    figure, axis = plt.subplots(figsize=(5.2, 2.03))
    for row, (variant, repeat, count) in enumerate(groups):
        values = [run["checkpoints"][-1]["one_step"]["disagreement_error_relation"]["spearman"]
                  for run in runs if (run["variant"], run["training_action_repeat"], run["training_count"]) == (variant, repeat, count)]
        axis.plot([min(values), max(values)], [row, row], color=COLORS[variant], linewidth=.7, alpha=.55)
        axis.scatter(values, np.full(len(values), row), s=15, color=COLORS[variant], zorder=3)
    axis.axvline(0, color="#a49b92", linewidth=.6, zorder=0)
    axis.set_yticks(range(len(groups)), [f"{variant} · {repeat} · {count:,}" for variant, repeat, count in groups], fontsize=8.2)
    axis.invert_yaxis()
    axis.set_xlim(-.23, .58)
    axis.set_xticks([-.2, 0, .2, .4, .6])
    axis.set_xlabel("Disagreement / actual error · Spearman correlation", fontsize=8.5)
    axis.tick_params(axis="y", length=0, pad=8)
    axis.tick_params(axis="x", labelsize=8, length=2.5, width=.4)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#888")
    axis.spines["bottom"].set_linewidth(.4)
    figure.subplots_adjust(left=.29, right=.98, top=.95, bottom=.23)
    return _svg(figure)


def build_report(data: dict) -> str:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "svg.fonttype": "none"})
    runs = data["runs"]
    expected = {(variant, repeat, count, seed) for variant in ("open", "rooms")
                for repeat in (1, 8) for count in (500, 2000) for seed in (0, 1, 2)}
    actual = {(r["variant"], r["training_action_repeat"], r["training_count"], r["seed"]) for r in runs}
    if actual != expected or len(runs) != 24 or any([c["fit_batches"] for c in r["checkpoints"]] != [0, 100, 500] for r in runs):
        raise ValueError("This companion requires the complete recorded 24-run, 0/100/500-batch study")
    groups = [(variant, repeat, count) for variant in ("open", "rooms") for repeat in (1, 8) for count in (500, 2000)]
    rows = []
    for variant, repeat, count in groups:
        selected = [r for r in runs if (r["variant"], r["training_action_repeat"], r["training_count"]) == (variant, repeat, count)]
        finals = [r["checkpoints"][-1] for r in selected]
        coverage = np.mean([r["training_data_summary"]["square_coverage_fraction"] for r in selected]) * 100
        errors = [np.mean([c["one_step"]["physical_mse"] for c in finals]) * 1e4,
                  np.mean([c["rollouts"]["8"]["endpoint_physical_mse"] for c in finals]) * 1e3,
                  np.mean([c["rollouts"]["24"]["endpoint_physical_mse"] for c in finals]) * 1e3]
        rows.append(f'<tr><td class="world {variant}">{variant}</td><td>{repeat}</td><td>{count:,}</td><td>{coverage:.1f}%</td>'
                    + "".join(f"<td>{value:.2f}</td>" for value in errors) + "</tr>")
    times = {b: np.median([next(c for c in r["checkpoints"] if c["fit_batches"] == b)["cumulative_training_seconds"] for r in runs]) for b in (100, 500)}
    correlations = [r["checkpoints"][-1]["one_step"]["disagreement_error_relation"]["spearman"] for r in runs]
    improved24 = sum(r["checkpoints"][-1]["rollouts"]["24"]["endpoint_physical_mse"] < r["checkpoints"][1]["rollouts"]["24"]["endpoint_physical_mse"] for r in runs)
    contact_notes = []
    for variant in ("open", "rooms"):
        selected = [r for r in runs if r["variant"] == variant]
        events = [r["checkpoints"][-1]["one_step"]["observed_velocity_drop_events"] for r in selected]
        frequency = np.mean([event["count"] / 1536 for event in events]) * 100
        error_share = np.mean([event["fraction_of_total_physical_squared_error"] for event in events]) * 100
        contact_notes.append(f'<div class="contact {variant}"><span>{variant}</span><strong>{frequency:.2f}% <i>→</i> {error_share:.0f}%</strong><small>transitions → total squared error</small></div>')
    source = data["provenance"]["source_sha256"]
    css = """
@page { size:A4 portrait; margin:11mm 14mm 10mm; }
* { box-sizing:border-box; }
html,body { margin:0; padding:0; }
body { color:#22211f; background:#fffff8; font:9.5pt/1.28 "Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
article { width:182mm; margin:0 auto; }
.page { break-after:page; page-break-after:always; }
.page:last-child { break-after:auto; page-break-after:auto; }
.kicker { margin:0 0 3mm; font-size:8.8pt; letter-spacing:.07em; text-transform:uppercase; }
h1 { margin:0 0 3mm; font-size:21pt; font-weight:normal; line-height:1.06; max-width:165mm; }
.deck { margin:0 0 6mm; font-size:10.2pt; color:#5b554e; }
h2 { font-weight:normal; font-style:italic; font-size:14pt; line-height:1.2; margin:0 0 2.2mm; }
p { margin:0 0 2.4mm; }
.row { display:grid; grid-template-columns:130mm 44mm; gap:8mm; margin-bottom:2mm; align-items:start; }
aside { font-size:8.5pt; line-height:1.28; color:#5a554e; padding-top:.6mm; }
aside p { margin-bottom:3mm; }
.label { font-size:8pt; text-transform:uppercase; letter-spacing:.05em; display:block; margin-bottom:1.5mm; color:#81766b; }
strong { font-weight:600; }
figure { margin:2mm 0 4mm; break-inside:avoid; }
figure svg { display:block; width:100%; height:auto; overflow:visible; }
figcaption { font-size:8.8pt; line-height:1.32; margin-top:2mm; color:#544d46; }
.full { margin:0 0 3mm; }
.full svg { max-height:48mm; }
.rule { border-top:.45pt solid #8e877e; padding-top:4mm; }
.open { color:#37657b; }
.rooms { color:#8b4b45; }
table { border-collapse:collapse; width:100%; margin-top:3mm; font-variant-numeric:tabular-nums; font-size:9.7pt; }
th { font-weight:normal; font-style:italic; border-top:.5pt solid #655e55; border-bottom:.45pt solid #655e55; }
th,td { text-align:right; padding:.85mm 1.4mm; }
th:first-child,td:first-child { text-align:left; padding-left:0; }
th small { font-style:normal; font-size:8.1pt; display:block; }
tr:nth-child(5) td { border-top:.3pt solid #c8c1b7; }
tbody tr:last-child td { border-bottom:.5pt solid #655e55; }
.table-note { font-size:8.5pt; color:#5a554e; margin:2mm 0 0; }
.contact { margin:1.5mm 0 2mm; border-left:1.5pt solid; padding-left:3mm; }
.contact span,.contact small { display:block; font-size:8.5pt; color:#5a554e; }
.contact strong { display:block; font-size:14pt; font-weight:normal; white-space:nowrap; }
.contact i { font-size:10pt; font-style:normal; color:#888; }
.mono { font:7.9pt/1.35 ui-monospace,Menlo,Consolas,monospace; overflow-wrap:anywhere; }
a { color:inherit; text-decoration:none; border-bottom:.4pt solid #a99a85; }
.reproduction { font-size:8.8pt; }
@media screen { body { padding:14mm 0; } .page { margin-bottom:18mm; } }
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Study 002 · Small world models learn motion before collisions</title><style>{css}</style></head><body><article>
<section class="page">
<div class="kicker">RTGA · Development study 002 · 5 September 2026</div>
<h1>Small world models learn motion<br>before collisions</h1>
<p class="deck">Random experience, short-horizon prediction, and the limits of ensemble agreement</p>
<div class="row"><section><h2>Abstract</h2><p>A three-member probabilistic ensemble learns useful motion predictions from 500–2,000 random-action transitions. In two small puck worlds, all 24 fitted models outperform constant-state prediction at one and eight steps; 20 do so at 24 steps. Residual error concentrates on rare contact-like transitions. Longer action holds increase spatial coverage, but coverage and ensemble agreement do not reliably establish accurate long-range dynamics. </p></section><aside><span class="label">Scope of evidence</span><p><strong>24 fits, three seeds per condition.</strong> This is a development diagnostic, with no confirmatory significance claim.</p><p>No goal, reward, pretrained model, imagined training examples, or task-directed collection.</p></aside></div>
<div class="row"><section><h2>Methods</h2><p>We cross two fixed layouts (<span class="open">open</span>, <span class="rooms">rooms</span>), uniform random action holds of 1 or 8 primitive frames, and training sets of 500 or 2,000 actual transitions. Smaller sets are prefixes of longer trajectories. Fitting budgets match; bootstrap batches are independent.</p><p>Each ensemble has three two-layer, 64-unit networks. Continuous heads predict Gaussian state changes; a Bernoulli head predicts the door flag. Checkpoints precede fitting and follow 100 and 500 optimizer batches. Evaluation uses 32 separate actual trajectories of 48 frames, from 16 initial positions, with both action-hold policies. Probe experience never enters replay.</p></section><aside><span class="label">What was held fixed</span><p>Adam 0.002; batch 128 per member; normalized state/delta scales [1,1,½,½,1,1]; log-variance bounds [−8,1]; binary loss weight 10.</p><p>Held-out starts differ from the collection start, but may overlap regions later visited. This is fixed-layout generalization.</p></aside></div>
<figure class="full">{_learning_figure(runs)}<figcaption><strong>Figure 1.</strong> Fitting reduces error sharply before 100 batches; another 400 updates do not uniformly improve long rollouts. Fine lines are individual runs, with means in blue (open) and red (rooms). Physical MSE averages x, y, v<sub>x</sub>, v<sub>y</sub> in declared observation units. The endpoint at 24 frames is 1.2 seconds ahead.</figcaption></figure>
<div class="row rule"><section><h2>The missing events matter</h2><p>Transitions in which a moving velocity coordinate suddenly reaches zero account for around 1% of probes, yet roughly 70% of the remaining physical squared error. Their mean per-event error is about 300–350 times that of other transitions. Small aggregate loss therefore conceals events that can invalidate plans near walls.</p><p>Long-rollout position RMSE remains 0.070–0.193 per coordinate in a unit-square world. Predictive improvement is evidence of learned motion; it is not evidence that the world is mastered.</p></section><aside><span class="label">Observed contact proxy</span>{''.join(contact_notes)}<p>Velocity magnitude &gt;0.025 becomes &lt;10<sup>−7</sup> on at least one axis. These are observed transitions, not oracle collision labels.</p></aside></div>
</section>
<section class="page">
<div class="kicker">Study 002 · Results and implications</div><h2 style="font-size:20pt;margin-bottom:4mm">Coverage is easier to obtain than reliable forecasts</h2>
<div class="full"><table><thead><tr><th>World</th><th>Hold<small>frames</small></th><th>Training<small>transitions</small></th><th>Coverage<small>256 cells</small></th><th>1-step MSE<small>×10⁻⁴</small></th><th>8-step MSE<small>×10⁻³</small></th><th>24-step MSE<small>×10⁻³</small></th></tr></thead><tbody>{''.join(rows)}</tbody></table><p class="table-note">Means over three seeds after 500 fitting batches. Spatial coverage includes all cells in the enclosing square. Holding an action repeats actual primitive interactions; it does not accelerate simulated time.</p></div>
<div class="row"><section><h2>Agreement is an imperfect signal</h2><p>Local disagreement and real one-step error have Spearman correlations from {min(correlations):.3f} to {max(correlations):.3f}; {sum(value < 0 for value in correlations)} of 24 are negative. The most-disagreed decile is error-enriched in the group means, but the ranking is inconsistent.</p><figure>{_uncertainty_figure(groups, runs)}<figcaption><strong>Figure 2.</strong> One dot per seed. Row labels give world, action hold, and training transitions. Members receive the same actual state/action when disagreement is computed. Lines connect the range across three seeds, not a confidence interval.</figcaption></figure></section><aside><span class="label">Calibration limits</span><p>These are error-ranking checks, not probability calibration or evidence of information gain. A separate experiment must test whether selected transitions produce subsequent learning progress.</p><span class="label">No mechanism learned</span><p>The door is always open in both layouts; the noise sensor is constant. Tiny Brier scores therefore test a constant flag. This study contains no switch event and cannot establish noisy-TV avoidance.</p></aside></div>
<div class="row rule"><section><h2>A bounded next experiment</h2><p>Median fitting time is <strong>{times[100]*1000:.0f} ms for 100 batches</strong> and <strong>{times[500]*1000:.0f} ms for 500</strong> on one CPU thread. The longer fit improves 24-step error in {improved24}/24 runs and worsens it in the others. Online comparisons should count warmup and fitting, and compare planning horizons without assuming an optimum.</p><p>Prioritize real-contact prediction, generic replay sampling of high observed errors, and closed-loop control checks. Keep the evaluation stream separate. Adding hidden exact-wall constraints would no longer test the learned simulator.</p></section><aside><span class="label">Provenance</span><p>Timing excludes collection and evaluation. Fits were not continued to convergence.</p><p>The constant-state baseline and contact breakdown were added descriptively after the initial run. Repeated prediction metrics matched exactly; model settings were unchanged.</p></aside></div>
<div class="row reproduction"><section><p><strong>Reproduce.</strong> <span class="mono">python -m rtga.model_study --output results/002-model-calibration</span>, then <span class="mono">python results/002-model-calibration/analyze.py</span>. Use the repository environment.</p><p><a href="../results/002-model-calibration/results.json">Complete records</a> include all configurations, per-transition errors, uncertainty values, and runtime provenance. <a href="../results/002-model-calibration/summary.json">Aggregates</a>, actual trajectories, 24 checkpoints, and exact source snapshots are retained. <a href="002-model-calibration.md">The longer note</a> states definitions and limitations.</p></section><aside><span class="label">Source SHA-256 prefixes</span><div class="mono">models {escape(source['models.py'][:12])}<br>envs {escape(source['envs.py'][:12])}<br>study {escape(source['model_study.py'][:12])}</div><p style="margin-top:2mm">Full hashes appear in the records.</p></aside></div>
</section></article></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/002-model-calibration/results.json"))
    parser.add_argument("--output", type=Path, default=Path("papers/002-model-calibration.html"))
    args = parser.parse_args()
    html = build_report(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
