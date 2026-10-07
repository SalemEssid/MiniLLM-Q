"""Phase 6 analysis of results/phase6_search.json: calibration perplexity of the parent against generation for
every run, and the spread over seeds of each width set and budget.

Writes results/phase6_analysis.json and two figures: results/phase6_convergence.png, and
results/phase6_allocation.png, the start and the result of the best seed of each width set and budget as a
grid of widths.
"""
import json
import math

import numpy as np
from matplotlib.ticker import MaxNLocator

from analyze_scan import DEPTH, INK, INK_2, KINDS, RESULTS, SURFACE, TYPE_COLORS, plt, save, style
from common import write_json


def main():
    search = json.loads((RESULTS / "phase6_search.json").read_text())
    phase5 = json.loads((RESULTS / "phase5_allocations.json").read_text())["allocations"]
    out = {"runs": {}, "configurations": {}}
    configs = {}
    for run_key, r in search["runs"].items():
        losses = [r["start_loss"]] + [g["loss"] for g in r["generations"]]
        out["runs"][run_key] = {
            "generations": len(r["generations"]), "start_perplexity": math.exp(losses[0]),
            "perplexity": math.exp(losses[-1]), "improvement": 1 - math.exp(losses[-1] - losses[0]),
            "generations_improved": sum(g["improved"] for g in r["generations"]),
            "tokens_scored": sum(g["tokens_scored"] for g in r["generations"]),
            "hours": sum(g["seconds"] for g in r["generations"]) / 3600, "trajectory": [math.exp(x) for x in losses]}
        configs.setdefault(f"{r['width_set']}/{r['budget']:g}", []).append(run_key)
    for config, keys in configs.items():
        final = [out["runs"][k]["perplexity"] for k in keys]
        greedy = phase5.get(f"{config}/greedy", {}).get("P")
        out["configurations"][config] = {"seeds": len(keys), "rd_start": out["runs"][keys[0]]["start_perplexity"],
                                         "greedy": greedy, "mean": float(np.mean(final)), "min": min(final),
                                         "max": max(final)}
    write_json(RESULTS / "phase6_analysis.json", out)

    # One panel per width set and budget; one line per seed (same colour for a seed in every panel).
    names = sorted(configs, key=lambda c: (c.split("/")[0], float(c.split("/")[1])))
    fig, axes = plt.subplots(1, len(names), figsize=(max(3.2 * len(names), 5.5), 3.4), squeeze=False)  # own y-axis each
    fig.patch.set_facecolor(SURFACE)
    axes = axes[0]
    for ax, config in zip(axes, names):
        style(ax)
        for k in sorted(configs[config]):
            seed = search["runs"][k]["seed"]
            t = out["runs"][k]["trajectory"]
            ax.step(range(len(t)), t, where="post", color=TYPE_COLORS[seed], linewidth=2, label=f"seed {seed}")
        ax.set_title(f"width set {config.split('/')[0]}, {config.split('/')[1]} bits", color=INK_2, fontsize=9)
        ax.set_xlabel("generation"), ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("calibration perplexity")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2)
    save(fig, "phase6_convergence.png", "Search: perplexity of the parent by generation")

    # Allocation maps: rows are module types, columns blocks, colour and number the width; outlined cells
    # are the ones the search changed.
    shade = {2: 0, 3: 3, 4: 6, 6: 9, 8: 11}  # steps of the one-hue ramp, light = few bits
    fig, axes = plt.subplots(len(names), 2, figsize=(10, 1.9 * len(names) + 0.4), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for row, config in zip(axes, names):
        best = min(configs[config], key=lambda k: out["runs"][k]["perplexity"])
        r = search["runs"][best]
        panels = [("rate-distortion start", r["start"], out["runs"][best]["start_perplexity"]),
                  (f"search, seed {r['seed']}", r["generations"][-1]["parent"], out["runs"][best]["perplexity"])]
        for ax, (label, widths, ppl) in zip(row, panels):
            ax.set_facecolor(SURFACE)
            for i, b in enumerate(widths):
                block, kind = divmod(i, len(KINDS))
                changed = label.startswith("search") and b != r["start"][i]
                ax.add_patch(plt.Rectangle((block + 0.04, kind + 0.04), 0.92, 0.92, color=DEPTH[shade[b]],
                                           ec=INK if changed else SURFACE, lw=2 if changed else 0))
                ax.text(block + 0.5, kind + 0.5, str(b), ha="center", va="center", fontsize=8,
                        color=SURFACE if shade[b] >= 5 else INK)
            ax.set_xlim(0, 12), ax.set_ylim(len(KINDS), 0)
            ax.set_xticks([b + 0.5 for b in range(12)], [str(b) for b in range(12)])
            ax.set_yticks([k + 0.5 for k in range(len(KINDS))], KINDS)
            ax.tick_params(length=0, colors=INK_2, labelsize=8)
            for side in ax.spines.values():
                side.set_visible(False)
            ax.set_title(f"width set {config.split('/')[0]}, {config.split('/')[1]} bits: {label}, "
                         f"perplexity {ppl:.1f}", color=INK_2, fontsize=9, loc="left")
        row[0].set_xlabel("block", color=INK_2, fontsize=8), row[1].set_xlabel("block", color=INK_2, fontsize=8)
    save(fig, "phase6_allocation.png", "Bits per module: outlined cells are the ones the search changed")

    print(json.dumps({k: {kk: v for kk, v in r.items() if kk != "trajectory"} for k, r in out["runs"].items()},
                     indent=2))
    print(json.dumps(out["configurations"], indent=2))


if __name__ == "__main__":
    main()
