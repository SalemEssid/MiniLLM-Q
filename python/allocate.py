"""Phase 5: closed-form allocation (guide, Section 8).

For each width set (A = {2, 4, 8}, B = {2, 3, 4, 6, 8}) and budget (an average of 3, 3.5 or 4 code bits over
the 48 block modules, weighted by size), three allocations:

  rd          rate-distortion, the main objective: minimize sum_l c_l 2^(-2 b_l), with c_l from Phase 4
  quadratic   MAPLE's quadratic moved to bits, the comparison: minimize sum_l s_l (b_l - b*_l)^2
  greedy      reference: the measured Phase 4 table alone, lowered greedily from all-8-bit

The closed forms are subject to sum_l n_l b_l = B. Each continuous solution is clamped to [2, 8], rounded to
the nearest width of the set, then repaired to the budget greedily with the measured table: while over
budget, lower the module with the smallest measured loss increase per byte saved; then, while a raise still
fits, raise the module with the largest loss decrease per byte added.

Every allocation is scored on the Phase 4 calibration slice (the test split is never used here), next to the
additive prediction sum_l dL_l(b_l). Uniform allocations, where the budget allows, come from Phase 4.

Writes results/phase5_allocations.json, saved after every evaluation, so a rerun resumes.
Usage: python allocate.py [--no-eval]
"""
import argparse
import json
import math

import numpy as np

from common import D, ROOT, disable_power_throttling, keep_awake, write_json
from quant import KINDS

WIDTH_SETS = {"A": [2, 4, 8], "B": [2, 3, 4, 6, 8]}
BUDGETS = [3.0, 3.5, 4.0]
METHODS = ["rd", "quadratic", "greedy"]
EPS = 1e-9  # floor for c_l and s_l, so no division by zero or log of zero
RESULTS = ROOT / "results"
OUT = RESULTS / "phase5_allocations.json"

# Parameters per module, in units of 768 x 768 = 589,824 (c_attn 3, attn.c_proj 1, c_fc 4, mlp.c_proj 4).
SIZE = {"attn.c_attn": 3, "attn.c_proj": 1, "mlp.c_fc": 4, "mlp.c_proj": 4}
N = np.array([SIZE[KINDS[i % len(KINDS)]] for i in range(48)], dtype=float)
UNIT = D * D


# ---- Closed forms: the answers to the Phase 5 derivation tasks (guide, appendices). Derive before reading. ----

def rd_continuous(c, n, bbar):
    log_cn = np.log2(np.maximum(c, EPS) / n)
    return bbar + 0.5 * (log_cn - np.sum(n / n.sum() * log_cn))


def quadratic_continuous(s, b_star, n, budget_bits):
    s = np.maximum(s, EPS)
    return b_star - (np.sum(n * b_star) - budget_bits) * (n / s) / np.sum(n ** 2 / s)

# ----------------------------------------------------------------------------------------------------------------


def to_width_set(b, widths):
    """Clamp to [2, 8] and round to the nearest width of the set (ties go to the narrower width)."""
    return [min(widths, key=lambda w: (abs(w - x), w)) for x in np.clip(b, 2, 8)]


def repair(alloc, widths, budget_bits, dL):
    """Greedy repair to the budget with the measured table dL[l][w]. Returns the allocation and the move count."""
    alloc, moves = list(alloc), 0
    used = lambda: float(np.dot(N, alloc))  # noqa: E731
    while used() > budget_bits + 1e-9:  # lower: smallest measured loss increase per byte saved
        options = [((dL[l][widths[widths.index(b) - 1]] - dL[l][b]) / (N[l] * (b - widths[widths.index(b) - 1])), l)
                   for l, b in enumerate(alloc) if b != widths[0]]
        _, l = min(options)
        alloc[l] = widths[widths.index(alloc[l]) - 1]
        moves += 1
    while True:  # raise, while one still fits: largest measured loss decrease per byte added
        room = budget_bits - used()
        options = [((dL[l][b] - dL[l][widths[widths.index(b) + 1]]) / (N[l] * (widths[widths.index(b) + 1] - b)), l)
                   for l, b in enumerate(alloc)
                   if b != widths[-1] and N[l] * (widths[widths.index(b) + 1] - b) <= room + 1e-9]
        if not options:
            return alloc, moves
        _, l = max(options)
        alloc[l] = widths[widths.index(alloc[l]) + 1]
        moves += 1


def allocations(analysis):
    """Every (width set, budget, method) allocation, with its continuous solution and repair record."""
    mods = analysis["modules"]
    c = np.array([m["c_alpha2"] for m in mods])
    s = np.array([m["s"] for m in mods])
    b_star = np.array([m["b_star"] for m in mods], dtype=float)
    dL = [{int(w): v for w, v in m["dL"].items()} for m in mods]
    out = {}
    for set_name, widths in WIDTH_SETS.items():
        for bbar in BUDGETS:
            budget_bits = bbar * N.sum()
            for method in METHODS:
                if method == "greedy":
                    continuous, rounded = None, [widths[-1]] * 48
                else:
                    continuous = (rd_continuous(c, N, bbar) if method == "rd"
                                  else quadratic_continuous(s, b_star, N, budget_bits))
                    rounded = to_width_set(continuous, widths)
                final, moves = repair(rounded, widths, budget_bits, dL)
                out[f"{set_name}/{bbar:g}/{method}"] = {
                    "width_set": set_name, "budget": bbar, "method": method, "widths": final,
                    "average_bits": float(np.dot(N, final) / N.sum()),
                    "predicted_dL": float(sum(dL[l][b] for l, b in enumerate(final))),
                    "continuous": None if continuous is None else [round(float(x), 3) for x in continuous],
                    "rounded_before_repair": rounded, "repair_moves": moves}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-eval", action="store_true", help="compute the allocations only")
    args = parser.parse_args()
    analysis = json.loads((RESULTS / "phase4_analysis.json").read_text())
    scan = json.loads((RESULTS / "phase4_scan.json").read_text())
    lnP0 = math.log(scan["P0"])

    state = json.loads(OUT.read_text()) if OUT.exists() else {"allocations": {}}
    measured = {tuple(a["widths"]): a["P"] for a in state["allocations"].values() if a.get("P") is not None}
    for key, a in scan.get("allocations_near", {}).items():  # the uniform allocations from Phase 4
        if key.endswith("uniform"):
            measured.setdefault(tuple(a["widths"]), a["P"])
    state.update({"calibration": scan["calibration"], "P0": scan["P0"], "module_units": N.tolist(),
                  "unit_parameters": UNIT, "width_sets": WIDTH_SETS, "budgets": BUDGETS})
    for key, a in allocations(analysis).items():
        a["P"] = measured.get(tuple(a["widths"]))
        state["allocations"][key] = a

    uniform = {}
    for set_name, widths in WIDTH_SETS.items():
        for bbar in BUDGETS:
            if bbar in widths:
                u = [int(bbar)] * 48
                uniform[f"{set_name}/{bbar:g}/uniform"] = {
                    "width_set": set_name, "budget": bbar, "method": "uniform", "widths": u, "average_bits": bbar,
                    "predicted_dL": float(sum(analysis["modules"][l]["dL"][str(b)] for l, b in enumerate(u))),
                    "P": measured.get(tuple(u))}
    state["allocations"].update(uniform)
    write_json(OUT, state)

    if not args.no_eval:
        print(f"power throttling disabled: {disable_power_throttling()}", flush=True)
        print(f"sleep blocked: {keep_awake()}", flush=True)
        from sensitivity_scan import Evaluator  # loads torch and the model only when evaluating
        ev = Evaluator()
        for key, a in state["allocations"].items():
            if a["P"] is None:
                widths = tuple(a["widths"])
                if widths not in measured:
                    measured[widths] = ev.perplexity(dict(enumerate(widths)))
                    print(f"{key}: perplexity {measured[widths]:.4f} ({ev.seconds[-1]:.0f} s)", flush=True)
                a["P"] = measured[widths]
                write_json(OUT, state)

    for a in state["allocations"].values():
        a["measured_dL"] = None if a["P"] is None else math.log(a["P"]) - lnP0
    write_json(OUT, state)
    print(f"\n{'allocation':24s} {'avg bits':>8s} {'predicted ppl':>14s} {'measured ppl':>13s}")
    for key in sorted(state["allocations"], key=lambda k: (k.split("/")[0], float(k.split("/")[1]), k)):
        a = state["allocations"][key]
        pred = scan["P0"] * math.exp(a["predicted_dL"])
        meas = "-" if a["P"] is None else f"{a['P']:.2f}"
        print(f"{key:24s} {a['average_bits']:8.3f} {pred:14.2f} {meas:>13s}")


if __name__ == "__main__":
    main()
