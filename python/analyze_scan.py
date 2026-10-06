"""Phase 4 analysis of results/phase4_scan.json (guide, Section 7).

  loss increase      dL_l(b) = ln P_l(b) - ln P_0
  rate-distortion    fit log2 dL_l(b) = log2 c_l - alpha_l * b over the widths with dL > 0; c_l with alpha
                     fixed at 2 is the input of the main closed form, the free alpha_l checks the model
  quadratic inputs   b*_l (smallest width within 0.5% of the 8-bit perplexity) and s_l (range of ln P_l),
                     with the ranking check that leaves b = 2 out
  interactions       pair ratio dL_lm / (dL_l + dL_m); for whole allocations, measured dL against the
                     additive prediction sum_l dL_l(b_l), with Spearman's rank correlation, separately for the
                     40-move set (far from uniform) and the 0-6-move set (near uniform)

Writes results/phase4_analysis.json and five figures, results/phase4_*.png.
"""
import json
import math

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from common import ROOT, write_json  # noqa: E402
from quant import KINDS, MODULES, WIDTHS  # noqa: E402

RESULTS = ROOT / "results"
TOLERANCE = 0.005

# Reference palette of the dataviz method, light mode: categorical slots 1-4 in fixed order for the four
# module types, one blue ramp (light = early block, dark = late block) for depth.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TYPE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DEPTH = ["#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def ranks(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):  # average ranks for ties
        r[x == v] = r[x == v].mean()
    return r


def spearman(a, b):
    return float(np.corrcoef(ranks(np.asarray(a)), ranks(np.asarray(b)))[0, 1])


def kendall(a, b):
    a, b = np.asarray(a), np.asarray(b)
    s = sum(np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]) for i in range(len(a)) for j in range(i + 1, len(a)))
    ta = sum(a[i] != a[j] for i in range(len(a)) for j in range(i + 1, len(a)))
    tb = sum(b[i] != b[j] for i in range(len(b)) for j in range(i + 1, len(b)))
    return float(s / math.sqrt(ta * tb))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(which="both", colors=INK_2, labelsize=8)
    ax.yaxis.label.set_color(INK_2), ax.xaxis.label.set_color(INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figure(width, height, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), sharey=ncols > 1, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes[0]:
        style(ax)
    return fig, axes[0]


def save(fig, name, title):
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plain_log_ticks(axis):
    """Plain numbers at 1, 2 and 5 of every decade on a log axis, instead of powers of ten."""
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def module_bars(values, ylabel, name, title):
    """One bar per module, grouped by block, coloured by type, on a log axis (values <= 0 are not drawn)."""
    fig, (ax,) = figure(10, 3.6)
    for i in range(48):
        block, kind = divmod(i, 4)
        ax.bar(block * 5 + kind, values[i], width=0.8, color=TYPE_COLORS[kind])
    ax.set_yscale("log"), ax.set_ylabel(ylabel), plain_log_ticks(ax.yaxis)
    ax.set_xticks([b * 5 + 1.5 for b in range(12)], [str(b) for b in range(12)]), ax.set_xlabel("block")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in TYPE_COLORS], labels=KINDS, frameon=False,
              fontsize=8, labelcolor=INK_2, ncols=4, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    save(fig, name, title)


def allocation_summary(items):
    """Spearman correlation and error of the additive prediction, over all items and per budget."""
    pr, me = [a["predicted"] for a in items], [a["measured"] for a in items]
    out = {"items": items, "spearman_all": spearman(pr, me),
           "median_measured_over_predicted": float(np.median(np.divide(me, pr))),
           "median_abs_error_nats": float(np.median(np.abs(np.subtract(me, pr))))}
    for budget in sorted({a["budget"] for a in items}):
        sub = [a for a in items if a["budget"] == budget]
        out[f"spearman_{budget}bit"] = spearman([a["predicted"] for a in sub], [a["measured"] for a in sub])
    return out


def main():
    scan = json.loads((RESULTS / "phase4_scan.json").read_text())
    lnP0 = math.log(scan["P0"])
    P = np.array([[scan["P"][str(i)][str(b)] for b in WIDTHS] for i in range(48)])  # [module, width]
    dL = np.log(P) - lnP0
    widths = np.array(WIDTHS)
    out = {"P0": scan["P0"], "modules": []}

    # Rate-distortion fit and the quadratic objective's inputs, per module.
    alphas, c2, s, s_no2, b_star = [], [], [], [], []
    for i in range(48):
        ok = dL[i] > 0
        y, x = np.log2(dL[i][ok]), widths[ok]
        alpha = float(-np.polyfit(x, y, 1)[0]) if ok.sum() >= 2 else float("nan")
        c = float(2 ** np.mean(y + 2 * x)) if ok.any() else 0.0
        lnP = np.log(P[i])
        b8 = P[i][WIDTHS.index(8)]
        bs = next(b for b, p in zip(WIDTHS, P[i]) if p <= (1 + TOLERANCE) * b8)
        alphas.append(alpha), c2.append(c), s.append(float(lnP.max() - lnP.min()))
        s_no2.append(float(lnP[1:].max() - lnP[1:].min())), b_star.append(bs)
        out["modules"].append({"index": i, "name": MODULES[i], "dL": dict(zip(map(str, WIDTHS), dL[i].tolist())),
                               "alpha": alpha, "c_alpha2": c, "b_star": bs, "s": s[-1], "s_without_2bit": s_no2[-1]})
    out["alpha"] = {"median": float(np.nanmedian(alphas)), "min": float(np.nanmin(alphas)),
                    "max": float(np.nanmax(alphas))}
    out["ranking_s_with_vs_without_2bit"] = {"kendall_tau": kendall(s, s_no2), "spearman": spearman(s, s_no2)}
    out["dL_3bit_share_of_top_5"] = float(np.sort(dL[:, 1])[-5:].sum() / dL[:, 1].sum())

    # Interactions.
    pair_bits = WIDTHS.index(3)
    strata = {}
    for key, p in scan.get("pairs", {}).items():
        a, b = map(int, key.split(","))
        ratio = (math.log(p["P"]) - lnP0) / (dL[a, pair_bits] + dL[b, pair_bits])
        strata.setdefault(p["stratum"], []).append(ratio)
    out["pairs"] = {k: {"n": len(v), "median_ratio": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}
                    for k, v in strata.items()}
    # Whole allocations, two sets: 40 random moves from uniform (far: many 2-bit modules) and 0-6 moves (near).
    sets = {}
    for name, field in [("far", "allocations"), ("near", "allocations_near")]:
        items = []
        for key, a in scan.get(field, {}).items():
            predicted = float(sum(dL[i, WIDTHS.index(b)] for i, b in enumerate(a["widths"])))
            items.append({"key": key, "budget": int(key.split("-")[0]), "predicted": predicted,
                          "measured": math.log(a["P"]) - lnP0})
        if items:
            sets[name] = items
            out[field] = allocation_summary(items)
    write_json(RESULTS / "phase4_analysis.json", out)

    # Figures 1 and 2: loss increase of every module at 3 bits, and its rate-distortion coefficient.
    module_bars(dL[:, pair_bits], "loss increase, nats", "phase4_loss_by_module.png",
                "Loss increase with one module at 3 bits, the rest FP32")
    module_bars(np.array(c2), "c_l, nats", "phase4_rd_coefficient.png",
                "Rate-distortion coefficient c_l (fit of dL = c_l 2^-2b)")

    # Figure 3: loss increase against width, one panel per type, one line per block (darker = deeper).
    fig, axes = figure(10, 3.4, ncols=4)
    for kind, ax in enumerate(axes):
        for block in range(12):
            i = 4 * block + kind
            ok = dL[i] > 0
            ax.plot(widths[ok], dL[i][ok], color=DEPTH[block], linewidth=2, marker="o", markersize=4)
        ref = np.median(dL[kind::4, WIDTHS.index(4)])
        ax.plot(widths, ref * 2.0 ** (-2 * (widths - 4)), color=INK_2, linewidth=1, linestyle="--")
        ax.set_yscale("log"), ax.set_xticks(WIDTHS), ax.set_title(KINDS[kind], color=INK_2, fontsize=9)
        ax.set_xlabel("bits")
    axes[0].set_ylabel("loss increase, nats")
    fig.text(0.99, 0.97, "dashed: slope 2^-2b    lighter line: earlier block", ha="right", va="top", fontsize=8,
             color=INK_2)
    save(fig, "phase4_loss_vs_bits.png", "Loss increase against width, per module")

    # Figure 4: additive prediction against measurement for whole allocations. Colour = budget; filled = near
    # uniform, hollow = 40 moves; the uniform allocations are labelled.
    if sets:
        fig, (ax,) = figure(5.2, 4.4)
        everything = [a for items in sets.values() for a in items]
        budgets = sorted({a["budget"] for a in everything})
        for name, items in sets.items():
            for k, budget in enumerate(budgets):
                sub = [a for a in items if a["budget"] == budget]
                if not sub:
                    continue
                style_kw = ({"color": TYPE_COLORS[k], "edgecolor": SURFACE, "linewidth": 1.5} if name == "near"
                            else {"facecolor": SURFACE, "edgecolor": TYPE_COLORS[k], "linewidth": 1.5})
                label = f"{budget}-bit, " + ("0-6 moves" if name == "near" else "40 moves")
                ax.scatter([a["predicted"] for a in sub], [a["measured"] for a in sub], s=40, label=label,
                           zorder=3, **style_kw)
                for a in sub:
                    if a["key"].endswith("uniform"):
                        ax.annotate("uniform", (a["predicted"], a["measured"]), xytext=(0, -14), ha="center",
                                    textcoords="offset points", fontsize=8, color=INK_2)
        pr, me = [a["predicted"] for a in everything], [a["measured"] for a in everything]
        lo = min(min(pr), min(me)) * 0.8
        hi = max(max(pr), max(me)) * 1.25
        ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle="--")
        ax.set_xscale("log"), ax.set_yscale("log")
        plain_log_ticks(ax.xaxis), plain_log_ticks(ax.yaxis)
        ax.set_xlabel("predicted: sum of single-module increases"), ax.set_ylabel("measured loss increase, nats")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
        save(fig, "phase4_additivity.png", "Do module errors add up?")

    # Figure 5: pair interaction ratios by stratum (1 = additive).
    if strata:
        fig, (ax,) = figure(7, 3.4)
        names = list(strata)
        rng = np.random.default_rng(0)
        for k, name in enumerate(names):
            v = strata[name]
            ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=30, color=TYPE_COLORS[0], edgecolor=SURFACE,
                       linewidth=1, zorder=3)
        ax.axhline(1, color=INK_2, linewidth=1, linestyle="--")
        ax.set_xticks(range(len(names)), [n.replace("/", "\n") for n in names], fontsize=8)
        ax.set_ylabel("pair increase / sum of singles")
        save(fig, "phase4_pairs.png", "Module pairs at 3 bits: 1 means the errors add up")

    summary = {k: v for k, v in out.items() if k != "modules"}
    for field in ("allocations", "allocations_near"):
        if field in summary:
            summary[field] = {k: v for k, v in summary[field].items() if k != "items"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
