"""One fixed equal-transition-budget follow-up to the rooms navigation failures."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .experiments import provenance, run_oracle, write_json
from .planning import PlannerConfig
from .viewer import write_viewer


LABELS = {"original": "Persistent, original", "immigrants": "Persistent, 10% immigrants",
          "repeat4": "Persistent, repeated initialization", "horizon48": "Persistent, H48 / N32",
          "fresh": "Fresh evolution"}
COLORS = {"original": "#8a1a1a", "immigrants": "#467b91", "repeat4": "#687b4e",
          "horizon48": "#956948", "fresh": "#786993"}
SEEDS = {"diagnostic": [108, 122], "development": list(range(10)), "confirmation": list(range(200, 220))}
STEPS = 80


def conditions():
    base = PlannerConfig(population=64, horizon=24, generations=4, method="persistent")
    return {"original": base, "immigrants": replace(base, immigrants=.10),
            "repeat4": replace(base, initial_repeat=4), "horizon48": replace(base, population=32, horizon=48),
            "fresh": replace(base, method="fresh")}


def core_hashes():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("envs.py", "planning.py", "experiments.py", "horizon_study.py")}


def design(output):
    output = Path(output)
    path = output / "design.json"
    protocol = {"experiment": "fixed_search_budget", "variant": "rooms", "steps": STEPS,
                "transitions_per_decision": 6144, "conditions": {k: asdict(v) for k, v in conditions().items()},
                "seeds": SEEDS, "success_radius": .06, "discount": .99,
                "confirmation_gate": "At least one persistent variant repairs an original failure on diagnostic seeds108/122, OR improves development success count with no increase in development mean distance. Run all five frozen conditions if the gate passes; no confirmation retuning.",
                "diagnostic_selection": "108 and122 were selected because original persistence failed in experiment001; they are not representative or held-out confirmation."}
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Existing design differs; do not silently replace the study protocol")
    write_json(path, protocol)
    return protocol


def trace_metrics(trace):
    frames = trace["frames"]
    states = np.asarray([f["state"] for f in frames])
    env = trace["environment"]
    contact_x = env["wall_x"] - env["wall_half_width"] - env["radius"]
    low, high = env["door_y"]
    at_wall = (np.abs(states[:, 0] - contact_x) < 1e-5) & (
        (states[:, 1] < low + env["radius"]) | (states[:, 1] > high - env["radius"]))
    longest = current = 0
    for hit in at_wall:
        current = current + 1 if hit else 0
        longest = max(longest, current)
    crossed = np.flatnonzero(states[:, 0] > env["wall_x"] + env["wall_half_width"] + env["radius"])
    entropy = []
    for f in frames:
        plans = np.asarray(f["plans"])
        if not plans.size:
            continue
        frequencies = np.bincount(plans[:, 0], minlength=6) / len(plans)
        entropy.append(float(-np.sum(frequencies[frequencies > 0] * np.log2(frequencies[frequencies > 0]))))
    return {"observed_wall_contact_frames": int(at_wall.sum()), "longest_wall_contact_run": longest,
            "first_observed_crossing_step": int(crossed[0]) if len(crossed) else None,
            "last_observed_position": states[-1, :2].tolist(),
            "mean_recorded_prefix_entropy_bits": float(np.mean(entropy)) if entropy else None,
            "prefix_entropy_scope": "First-action entropy of the24 stored, score-ordered plans, not the complete population",
            "observation_timing": "Pre-action states0..79; final post-action state80 is not in these trace diagnostics"}


def run_phase(output, phase):
    output = Path(output)
    protocol = design(output)
    if phase == "confirmation" and not confirmation_gate(output)["advance"]:
        raise ValueError("The fixed development gate did not warrant confirmation")
    result_path = output / phase / "results.json"
    if result_path.exists():
        data = json.loads(result_path.read_text())
        if data["complete"]:
            return data
        if data["core_sha256"] != core_hashes():
            raise ValueError("Source changed since an incomplete stage; do not mix source versions")
    else:
        data = {"experiment": "search_budget_followup", "phase": phase, "complete": False,
                "provenance": provenance(), "core_sha256": core_hashes(), "design": protocol, "records": []}
        write_json(result_path, data)
    existing = {(r["seed"], r["condition"]) for r in data["records"]}
    for seed in SEEDS[phase]:
        paired_start = paired_goal = None
        for name, config in conditions().items():
            if (seed, name) in existing:
                continue
            capture = phase == "diagnostic" or (phase == "confirmation" and seed in {200, 201})
            record, trace = run_oracle(seed, config.method, variant="rooms", steps=STEPS,
                                       planner_config=config, capture_trace=capture)
            if record["model_transitions"] != STEPS * 6144:
                raise AssertionError("Unequal hypothetical-transition budget")
            if paired_start is not None:
                assert record["start"] == paired_start and record["goal"] == paired_goal
            paired_start, paired_goal = record["start"], record["goal"]
            record.update(condition=name, phase=phase)
            if capture:
                stem = f"trace-seed{seed}-{name}"
                record["trace_diagnostics"] = trace_metrics(trace)
                trace["metadata"].update(title=f"Search budget / {phase} / {name} / seed{seed}",
                                          condition=name, phase=phase)
                trace["metadata"]["metrics"].update(record["trace_diagnostics"])
                result_path.parent.mkdir(parents=True, exist_ok=True)
                # Compact lossless archives keep all recorded plans and paths.
                trace_bytes = json.dumps(trace, separators=(",", ":"), allow_nan=False).encode()
                (result_path.parent / f"{stem}.json.gz").write_bytes(gzip.compress(trace_bytes, mtime=0))
                write_viewer(trace, result_path.parent / f"{stem}.html")
            data["records"].append(record)
            write_json(result_path, data)
            print(json.dumps({k: record[k] for k in ("phase", "condition", "seed", "success", "first_success_step", "mean_distance", "final_distance")}), flush=True)
    if data["core_sha256"] != core_hashes():
        raise ValueError("Source changed during the study stage; results are not marked complete")
    data["complete"] = True
    write_json(result_path, data)
    return data


def summaries(records):
    result = {}
    for name in conditions():
        rows = sorted((r for r in records if r["condition"] == name), key=lambda r: r["seed"])
        if not rows:
            continue
        result[name] = {"n": len(rows), "successes": sum(r["success"] for r in rows),
                        "failed_seeds": [r["seed"] for r in rows if not r["success"]],
                        "mean_distance": float(np.mean([r["mean_distance"] for r in rows])),
                        "final_distance": float(np.mean([r["final_distance"] for r in rows])),
                        "restricted_hit": float(np.mean([r["first_success_step"] or STEPS for r in rows])),
                        "latency_p50_ms": float(np.percentile(np.concatenate([r["latency_ms"] for r in rows]), 50)),
                        "latency_p95_ms": float(np.percentile(np.concatenate([r["latency_ms"] for r in rows]), 95))}
    return result


def confirmation_gate(output):
    output = Path(output)
    diagnostic = json.loads((output / "diagnostic/results.json").read_text())
    development_path = output / "development/results.json"
    if not diagnostic["complete"]:
        raise ValueError("Diagnostic stage incomplete")
    if not development_path.exists():
        raise ValueError("Complete development seeds0..9 before confirmation")
    development = json.loads(development_path.read_text())
    if not development["complete"]:
        raise ValueError("Development stage incomplete")
    d = {(r["seed"], r["condition"]): r for r in diagnostic["records"]}
    s = summaries(development["records"])
    repairs = {name: [seed for seed in SEEDS["diagnostic"] if not d[seed,"original"]["success"] and d[seed,name]["success"]]
               for name in ("immigrants", "repeat4", "horizon48")}
    development_improvements = [name for name in repairs if s[name]["successes"] > s["original"]["successes"]
                                and s[name]["mean_distance"] <= s["original"]["mean_distance"]]
    gate = {"advance": any(repairs.values()) or bool(development_improvements), "diagnostic_repairs": repairs,
            "development_improvements": development_improvements,
            "decision": "Run all five fixed conditions on200..219 without retuning"}
    write_json(output / "confirmation-gate.json", gate)
    return gate


def paired_effects(records):
    index = {(r["seed"], r["condition"]): r for r in records}
    seeds = sorted({r["seed"] for r in records})
    rng = np.random.default_rng(2026090504)
    sample = rng.integers(len(seeds), size=(20000, len(seeds)))
    effects = {}
    for name in conditions():
        if name == "original":
            continue
        result = {}
        for metric in ("success", "mean_distance", "final_distance", "restricted_hit"):
            def value(r):
                return (r["first_success_step"] or STEPS) if metric == "restricted_hit" else float(r[metric])
            values = np.array([value(index[s,name]) - value(index[s,"original"]) for s in seeds])
            lo, hi = np.percentile(values[sample].mean(axis=1), [2.5, 97.5])
            result[metric] = [float(values.mean()), float(lo), float(hi)]
        effects[name] = result
    return effects


def diagnostic_figure(output):
    output = Path(output)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rtga-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(2, 5, figsize=(12.2, 5.3), sharex=True, sharey=True)
    for row, seed in enumerate(SEEDS["diagnostic"]):
        for col, name in enumerate(conditions()):
            trace = json.loads(gzip.decompress((output / "diagnostic" / f"trace-seed{seed}-{name}.json.gz").read_bytes()))
            ax = axes[row,col]
            env = trace["environment"]
            for low, high in ((0, env["door_y"][0]), (env["door_y"][1], 1)):
                ax.add_patch(Rectangle((env["wall_x"]-env["wall_half_width"], low),2*env["wall_half_width"],high-low,color="#bbbbbb"))
            path = np.array([f["state"] for f in trace["frames"]])
            ax.plot(path[:,0],path[:,1],color=COLORS[name],linewidth=1.6)
            ax.plot(*path[0,:2],marker="o",color=COLORS[name],markersize=3)
            ax.plot(*env["goal"],marker="x",color="#333333",markersize=6)
            ax.set(xlim=(0,1),ylim=(0,1),aspect="equal",xticks=[0,.5,1],yticks=[0,.5,1])
            ax.tick_params(labelsize=8)
            success = trace["metadata"]["metrics"]["success"]
            ax.set_title(f"{name} · {'hit' if success else 'miss'}",fontsize=9,color=COLORS[name])
            if col==0:ax.set_ylabel(f"seed {seed}\ny",fontsize=9)
            if row==1:ax.set_xlabel("x",fontsize=9)
    fig.suptitle("Selected diagnostic failures · observed paths before actions0–79; crosses mark goals",fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "diagnostic-paths.png",dpi=170)
    fig.savefig(output / "diagnostic-paths.svg")
    plt.close(fig)


def write_report(output, paper):
    output, paper = Path(output), Path(paper)
    stages = {phase: json.loads((output / phase / "results.json").read_text()) for phase in SEEDS
              if (output / phase / "results.json").exists()}
    if any(not data["complete"] for data in stages.values()):
        raise ValueError("Finish all launched stages before reporting")
    report = ["# 004 · Escaping a wall trap under a fixed simulation budget", "",
              "A fixed exploratory follow-up to experiment001. Exact dynamics, rooms environment, dense distance objective; no learning, curiosity reward, waypoints or changes to core planning/physics.", "",
              "## Question and design", "",
              "Does persistence preserve a poor route, and can fresh proposals or more lookahead repair it? Every condition uses6,144 hypothetical transitions per decision and80 real actions (491,520 transitions/run). Seeds108 and122 were deliberately selected from the earlier failures and are diagnostic, not representative. Seeds0–9 are development; seeds200–219, if present below, are a separate fixed confirmation set.", "",
              "| Condition | Population × horizon × generations | Exact change |", "| --- | --- | --- |",
              "| Original persistence | 64 ×24 ×4 | Entire shifted population retained; random one-action tails. |",
              "| 10% immigrants | 64 ×24 ×4 | Six random genomes replace non-elites after each of three offspring generations; no extra per-tick reset. |",
              "| Repeated initialization | 64 ×24 ×4 | Four repeated primitive actions per sampled chunk in the first population only. Subsequent appended tails remain independent single actions. |",
              "| H48 /N32 | 32 ×48 ×4 | Twice the horizon and half the population; default mutation probability changes from1/24 to1/48, and elite count from6 to3. |",
              "| Fresh reference | 64 ×24 ×4 | Fresh population each decision, otherwise original genetic operators. |", "",
              "Equal transition counts are not equal wall time: doubling sequential rollout length can cost more despite halving batch width. Longer horizons also change the averaging window of the same discounted-distance objective. This experiment tests these concrete configurations, not an isolated causal horizon effect.", ""]
    for phase, data in stages.items():
        s=summaries(data["records"])
        report += [f"## {phase.capitalize()} outcomes", "", "| Condition | Success | Mean distance | Final distance | Restricted hit | Latency p50 /p95 ms | Failed seeds |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for name,r in s.items():
            report.append(f"| {LABELS[name]} | {r['successes']}/{r['n']} | {r['mean_distance']:.4f} | {r['final_distance']:.4f} | {r['restricted_hit']:.1f} | {r['latency_p50_ms']:.2f} /{r['latency_p95_ms']:.2f} | {', '.join(map(str,r['failed_seeds'])) or 'none'} |")
        report += ["", f"[Raw outcomes]({os.path.relpath(output / phase / 'results.json', paper.parent)}) include every run and distance/action/latency sequence.", ""]
    if "development" in stages:
        gate=confirmation_gate(output)
        report += ["## Frozen advancement decision", "", f"Gate passed: {gate['advance']}. Diagnostic repairs: `{gate['diagnostic_repairs']}`. Development improvements: `{gate['development_improvements']}`.", "", "The advancement rule was stored before data collection. It required a persistent variant to repair a selected original failure, or improve development success without increasing mean distance. If passed, all five conditions were retained for one confirmation comparison; no tuning used confirmation outcomes.", ""]
    if "confirmation" in stages:
        effects=paired_effects(stages["confirmation"]["records"])
        write_json(output / "confirmation" / "paired-effects.json", effects)
        report += ["## Paired confirmation effects", "", "Effects are condition minus original persistence, paired by environment seed. 95% percentile intervals use20,000 resamples of20 seed pairs, NumPy seed2026090504. Positive success differences favor the variant; negative distance/time differences favor the variant. Intervals are unadjusted for multiple comparisons. Restricted hit is min(first-success step,80), with failed runs contributing80; it is not a mean over successful runs alone.", "", "| Condition | Δ success, pp | Δ mean distance | Δ final distance | Δ restricted hit |", "| --- | --- | --- | --- | --- |"]
        for name,values in effects.items():
            cells=[]
            for metric in ("success","mean_distance","final_distance","restricted_hit"):
                factor,digits=(100,1) if metric=="success" else (1,1) if metric=="restricted_hit" else (1,4)
                cells.append(f"{values[metric][0]*factor:+.{digits}f} [{values[metric][1]*factor:+.{digits}f}, {values[metric][2]*factor:+.{digits}f}]")
            report.append("| "+LABELS[name]+" | "+" | ".join(cells)+" |")
        report += [""]
    report += ["## Diagnostic paths and what they can establish", "", f"![Selected diagnostic paths]({os.path.relpath(output / 'diagnostic-paths.png',paper.parent)})", "",
               "These are actual pre-action paths, with goals marked by crosses. A path stalled beside the wall shows an observed failure to find the doorway within80 actions; it does not by itself prove genetic convergence or memory as the sole cause. The unchanged perfect simulator removes learned-model error as an explanation. Stored candidate paths allow the short-horizon prediction to be compared with the executed trail.", "",
               "| Seed /condition | Longest wall-contact run | First observed crossing | Last observed position |", "| --- | --- | --- | --- |"]
    for r in stages["diagnostic"]["records"]:
        d=r["trace_diagnostics"]
        report.append(f"| {r['seed']} /{r['condition']} | {d['longest_wall_contact_run']} | {d['first_observed_crossing_step']} | {d['last_observed_position']} |")
    report += ["", "Wall contact uses the observed puck center at the collision boundary outside the traversable doorway. Trace diagnostics cover pre-action states0–79; the outcome table includes the final80th transition. Entropy in the raw diagnostics describes only the24 stored score-ordered genomes, not full-population diversity. Compressed `.json.gz` traces and standalone HTML viewers are stored for both diagnostic seeds and preselected confirmation seeds200/201.", "", "## Provenance and limits", "",
               "Each phase records its starting Git revision/dirty status, actual Python invocation and SHA-256 hashes of the exact runner, evaluator, planner and physics source. Other agents may change unrelated files concurrently; the runner refuses to mark a phase complete if its numerical sources change. These later runs do not modify experiment001. Timing is descriptive on a shared machine and includes no learning or environment/trace I/O.", ""]
    for phase,data in stages.items():
        prov=data["provenance"]
        report += [f"### {phase}", "", f"Run commit `{prov['git_commit']}`; dirty at start: `{prov['git_dirty']}`. A dirty checkout is identified by source hashes, not the commit alone.", "", "```json", json.dumps({"command":prov["command"],"core_sha256":data["core_sha256"]},indent=2), "```", ""]
    paper.parent.mkdir(parents=True,exist_ok=True)
    paper.write_text("\n".join(report)+"\n")
    return paper


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("results/004-search-budget"))
    p.add_argument("--phase",choices=[*SEEDS,"report"],required=True)
    p.add_argument("--paper",type=Path,default=Path("papers/004-search-budget.md"))
    args=p.parse_args()
    if args.phase=="report":
        diagnostic_figure(args.output)
        print(write_report(args.output,args.paper))
    else:
        run_phase(args.output,args.phase)


if __name__=="__main__":
    main()
