"""Render fixed-study spatial diagnostics; does not fit or alter models."""

from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


ROOT = Path(__file__).resolve().parent
results = json.loads((ROOT / 'results.json').read_text())
probes = np.load(ROOT / 'grid-probes.npz')
event = np.asarray(results['event']['state'])


def panel_values(candidate, seed):
    filename = ('knn-predictions.npz' if candidate == 'knn' else
                f'{candidate}-seed{seed}-updates3000-predictions.npz')
    with np.load(ROOT / filename) as archive:
        return archive['grid_mean_members'].mean(axis=0)[:, 4].reshape(13, 13, 6, 6)


def draw_map(ax, values, title, velocity, show_ylabel):
    probability = values[:, :, velocity, 5].T
    im = ax.imshow(probability, origin='lower', extent=(.17, .43, .37, .63),
                   vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
    ax.add_patch(Circle((.3, .5), .09, fill=False, edgecolor='black',
                        linestyle='--', linewidth=1.1))
    xs, ys = np.meshgrid(np.linspace(.18, .42, 13), np.linspace(.38, .62, 13))
    false = (probability >= .5) & (np.hypot(xs - .3, ys - .5) > .09)
    ax.scatter(xs[false], ys[false], s=15, marker='x', linewidth=.8, color='#ff4f57')
    ax.scatter([event[0]], [event[1]], s=85, marker='*', color='white',
               edgecolor='black', linewidth=.8, zorder=4)
    ax.set(title=title, xlabel='x position', xticks=[.2, .3, .4], yticks=[.4, .5, .6])
    ax.set_title(title, fontsize=10)
    if show_ylabel:
        ax.set_ylabel('y position')
    return im


fig, axes = plt.subplots(2, 3, figsize=(11.5, 8))
fig.subplots_adjust(left=.07, right=.85, bottom=.17, top=.84, wspace=.36, hspace=.60)
for column, candidate in enumerate(['uniform', 'high_change', 'knn']):
    fits = [panel_values(candidate, seed) for seed in ([None] if candidate == 'knn' else [0, 1, 2])]
    values = np.mean(fits, axis=0)
    rec = [r for r in results['records'] if r['candidate'] == candidate
           and r['optimizer_updates'] in [0, 3000]]
    p = np.mean([r['event_probability_mean'] for r in rec])
    title = {'uniform': 'Uniform neural replay', 'high_change': 'High-change neural replay',
             'knn': '5-neighbour memory'}[candidate]
    for row, velocity in enumerate([0, 5]):
        subtitle = 'Zero velocity' if row == 0 else 'Experienced velocity'
        im = draw_map(axes[row, column], values,
                      f'{title}\nExact event: {p:.2%} · {subtitle}', velocity, column == 0)
colorbar_ax = fig.add_axes([.885, .24, .018, .53])
fig.colorbar(im, cax=colorbar_ax, label='Predicted opening probability on interact')
fig.suptitle('Remembering one opening does not identify where interaction works', fontsize=15, y=.97)
fig.text(.5, .045,
         'Dashed circle: true switch region. White star: experienced event. Red crosses: false openings at p ≥ 0.5.\n'
         'Neural panels average three seeded fits at 3,000 updates; all panels use the same unchanged 2,500 transitions.',
         ha='center', fontsize=10)
fig.savefig(ROOT / 'spatial-generalization.png', dpi=170, bbox_inches='tight')
plt.close(fig)

# Retain each fitted model's spatial map, not only an average across fits.
for candidate in ['uniform', 'high_change']:
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), layout='constrained')
    for seed in [0, 1, 2]:
        values = panel_values(candidate, seed)
        for row, velocity in enumerate([0, 5]):
            im = draw_map(axes[row, seed], values,
                          f'Seed {seed} · {"zero" if row == 0 else "experienced"} velocity',
                          velocity, seed == 0)
    fig.colorbar(im, ax=axes, label='Predicted opening probability on interact', shrink=.85)
    fig.suptitle(f'{candidate}: each 3,000-update fit', fontsize=15)
    fig.savefig(ROOT / f'{candidate}-spatial-by-seed.png', dpi=150)
    plt.close(fig)

summary = {}
for candidate in ['uniform', 'high_change', 'knn']:
    rows = [r for r in results['records'] if r['candidate'] == candidate
            and r['optimizer_updates'] in [0, 3000]]
    nav = [r for r in results['navigation'] if r['candidate'] == candidate]
    summary[candidate] = {
        'event_p_mean': float(np.mean([r['event_probability_mean'] for r in rows])),
        'event_p_range': [min(r['event_probability_mean'] for r in rows),
                          max(r['event_probability_mean'] for r in rows)],
        'opening_recall_mean': float(np.mean([r['grid_all']['recall_at_half'] for r in rows])),
        'outside_interact_false_positive_mean': float(np.mean([
            r['grid_outside_interact']['false_positive_rate_at_half'] for r in rows])),
        'continuous_normalized_mse_mean': float(np.mean([
            r['motion']['continuous_normalized_mse'] for r in rows])),
        'navigation_trials': len(nav),
        'navigation_successes': sum(r['success'] for r in nav),
        'default_start_trials': sum(r['start'] == 'default' for r in nav),
        'default_start_successes': sum(r['success'] and r['start'] == 'default' for r in nav),
    }
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
