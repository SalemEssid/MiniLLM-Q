"""Phase 6: evolutionary search over allocations, after EvoPress (Sieberling et al., 2025) (guide, Section 9).

One run is one width set, one budget and one seed. It starts from the Phase 5 rate-distortion allocation,
and every generation:

  1. makes 16 offspring of the parent, each with 1-3 byte-neutral mutations: one module moves one width step
     of the set up or down, and another module of the same type (so the same size) moves the same number of
     bits the other way;
  2. scores them on 2 random calibration windows (2K tokens) and keeps the best 4;
  3. scores those on 8 random windows (8K tokens) and keeps the best 2;
  4. scores those on all 16 windows (16K tokens); the better of the best one and the parent is the next parent.

Fitness is the mean next-token cross-entropy on the Phase 4 calibration slice (EvoPress uses the KL divergence
to the FP32 model instead). The test split is never used here.

The NLL of every window is cached by allocation, so repeated candidates and overlapping windows cost nothing;
the cache is shared by all runs. State is saved after every generation, and generation g always draws the same
candidates (its random stream depends only on the run and g), so a rerun resumes exactly.

Writes results/phase6_search.json.
Usage: python search.py --set B --budget 3 --seed 0 [--generations 20]
       python search.py --all [--generations 20]      # sets A, B x budgets 3, 3.5 x seeds 0, 1, 2
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch

from allocate import N, WIDTH_SETS
from common import DATA, ROOT, disable_power_throttling, keep_awake, load_model, read_weights, versions, window_nll, \
    write_json
from quant import MODULES, conv1d, fake_quant
from sensitivity_scan import CALIB_TOKENS, WINDOW

OUT = ROOT / "results" / "phase6_search.json"
OFFSPRING, MAX_MUTATIONS = 16, 3
N_WINDOWS = CALIB_TOKENS // WINDOW
STAGES = [(2, 4), (8, 2), (N_WINDOWS, 1)]  # (windows scored, survivors)
ALL_RUNS = [(s, b, seed) for s in ("A", "B") for b in (3.0, 3.5) for seed in (0, 1, 2)]


def key(alloc):
    return "".join(map(str, alloc))  # every width is one digit


def mutate(alloc, widths, rng):
    """One byte-neutral move, or None when the chosen module has no partner that can move the other way."""
    a = rng.randrange(48)
    i = widths.index(alloc[a])
    new_a = widths[rng.choice([j for j in (i - 1, i + 1) if 0 <= j < len(widths)])]
    delta = new_a - alloc[a]
    partners = [b for b in range(a % 4, 48, 4) if b != a and alloc[b] - delta in widths]
    if not partners:
        return None
    b = rng.choice(partners)
    out = list(alloc)
    out[a], out[b] = new_a, alloc[b] - delta
    return out


def offspring(parent, widths, rng):
    """OFFSPRING distinct children of parent, each 1 to MAX_MUTATIONS moves away."""
    kids, seen = [], {key(parent)}
    for _ in range(100 * OFFSPRING):
        child = parent
        for _ in range(rng.randint(1, MAX_MUTATIONS)):
            child = mutate(child, widths, rng) or child
        if key(child) not in seen:
            assert np.dot(N, child) == np.dot(N, parent)
            kids.append(child), seen.add(key(child))
            if len(kids) == OFFSPRING:
                break
    return kids


class Scorer:
    """Calibration loss of allocations, window by window, with the cache. Only modules whose width changed
    are re-quantized between candidates."""

    def __init__(self, cache):
        self.model = load_model()
        tokens = np.fromfile(DATA / "wikitext2_train.u16", dtype="<u2")[:CALIB_TOKENS].astype(np.int64)
        self.windows = torch.from_numpy(tokens).view(N_WINDOWS, WINDOW)
        self.file = read_weights(DATA / "gpt2_124M.bin")  # memory-mapped FP32 originals
        self.loaded = [None] * 48
        self.cache = cache
        self.tokens_scored = 0

    def load(self, alloc):
        with torch.inference_mode():
            for i, b in enumerate(alloc):
                if self.loaded[i] != b:
                    W = torch.from_numpy(np.array(self.file[MODULES[i] + ".weight"]))
                    conv1d(self.model, MODULES[i]).weight.copy_(fake_quant(W, b))
                    self.loaded[i] = b

    def loss(self, alloc, windows):
        """Mean NLL of alloc over the given window indices."""
        entry = self.cache.setdefault(key(alloc), {})
        todo = [w for w in windows if str(w) not in entry]
        if todo:
            self.load(alloc)
            nll = window_nll(self.model, self.windows[todo].reshape(-1), WINDOW)
            entry.update({str(w): v for w, v in zip(todo, nll)})
            self.tokens_scored += len(todo) * WINDOW
        return float(np.mean([entry[str(w)] for w in windows]))


def run(scorer, state, start, set_name, budget, seed, generations):
    widths, run_key = WIDTH_SETS[set_name], f"{set_name}/{budget:g}/s{seed}"
    all_windows = list(range(N_WINDOWS))
    r = state["runs"].setdefault(run_key, {"width_set": set_name, "budget": budget, "seed": seed, "start": start,
                                           "start_loss": scorer.loss(start, all_windows), "generations": []})
    parent = r["generations"][-1]["parent"] if r["generations"] else start
    parent_loss = scorer.loss(parent, all_windows)
    print(f"{run_key}: start perplexity {math.exp(r['start_loss']):.3f}", flush=True)
    for g in range(len(r["generations"]) + 1, generations + 1):
        rng = random.Random(f"{run_key}/{g}")
        t0, tokens0 = time.perf_counter(), scorer.tokens_scored
        pool = offspring(parent, widths, rng)
        for n_windows, survivors in STAGES:
            ws = all_windows if n_windows == N_WINDOWS else sorted(rng.sample(all_windows, n_windows))
            pool = sorted(pool, key=lambda a: scorer.loss(a, ws))[:survivors]
        best, best_loss = pool[0], scorer.loss(pool[0], all_windows)
        improved = best_loss < parent_loss
        if improved:
            parent, parent_loss = best, best_loss
        r["generations"].append({"generation": g, "parent": parent, "loss": parent_loss, "improved": improved,
                                 "best_offspring_loss": best_loss, "tokens_scored": scorer.tokens_scored - tokens0,
                                 "seconds": time.perf_counter() - t0})
        write_json(OUT, state)
        print(f"{run_key} gen {g:2d}: perplexity {math.exp(parent_loss):.3f}"
              f"{' (improved)' if improved else ''}, best offspring {math.exp(best_loss):.3f}, "
              f"{(scorer.tokens_scored - tokens0) // 1024}K tokens, {time.perf_counter() - t0:.0f} s", flush=True)
    r["final"] = {"widths": parent, "loss": parent_loss, "perplexity": math.exp(parent_loss)}
    write_json(OUT, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=list(WIDTH_SETS))
    parser.add_argument("--budget", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="all 12 runs, one after another")
    parser.add_argument("--generations", type=int, default=20)
    args = parser.parse_args()
    runs = ALL_RUNS if args.all else [(args.set, args.budget, args.seed)]
    if not args.all and (args.set is None or args.budget is None):
        parser.error("give --set and --budget, or --all")

    print(f"power throttling disabled: {disable_power_throttling()}", flush=True)
    print(f"sleep blocked: {keep_awake()}", flush=True)
    phase5 = json.loads((ROOT / "results" / "phase5_allocations.json").read_text())["allocations"]
    state = json.loads(OUT.read_text()) if OUT.exists() else {"runs": {}, "cache": {}}
    state.update({"fitness": "mean next-token NLL on the calibration slice", "calibration": {
        "tokens": f"wikitext2_train.u16[0:{CALIB_TOKENS}]", "windows": N_WINDOWS, "window": WINDOW},
        "offspring": OFFSPRING, "max_mutations": MAX_MUTATIONS, "stages": STAGES,
        "torch_threads": torch.get_num_threads(), "versions": versions()})
    scorer = Scorer(state["cache"])
    for set_name, budget, seed in runs:
        run(scorer, state, phase5[f"{set_name}/{budget:g}/rd"]["widths"], set_name, budget, seed, args.generations)


if __name__ == "__main__":
    main()
