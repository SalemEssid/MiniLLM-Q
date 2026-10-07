"""Phase 6 analysis of results/phase6_search.json: calibration perplexity of the parent against generation for
every run, and the spread over seeds of each width set and budget.

Writes results/phase6_analysis.json and results/phase6_convergence.png.
"""
import json
import math

import numpy as np

from analyze_scan import INK_2, RESULTS, SURFACE, TYPE_COLORS, plt, save, style
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
        ax.set_xlabel("generation")
    axes[0].set_ylabel("calibration perplexity")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2)
    save(fig, "phase6_convergence.png", "Search: perplexity of the parent by generation")

    print(json.dumps({k: {kk: v for kk, v in r.items() if kk != "trajectory"} for k, r in out["runs"].items()},
                     indent=2))
    print(json.dumps(out["configurations"], indent=2))


if __name__ == "__main__":
    main()
