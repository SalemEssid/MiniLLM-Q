"""Phase 4: single-module sensitivity scan, then the interaction tests (guide, Section 7).

Everything is measured on the calibration slice: the first 16,384 tokens of the WikiText-2 train split, as
16 non-overlapping windows of 1024 (the test split is never touched here). Quantization is fake_quant, the
exact format of the C++ kernels.

  scan          for every block module l and width b in {2, 3, 4, 6, 8}: only module l quantized, the rest
                FP32, perplexity P_l(b); plus the FP32 perplexity P_0. 241 evaluations.
  pairs         60 module pairs at 3 bits, stratified by distance and type (same block, adjacent, distant;
                same type, different type).
  allocations   10 random allocations at an average of 3 code bits and 10 at 4, made by random byte-neutral
                moves from the uniform allocation.
  near          the same test close to uniform, where Phase 5's allocations live: per budget, the uniform
                allocation and 9 random ones made with only 2, 4 or 6 moves. The 40-move allocations above all
                put 6-16 modules at 2 bits (perplexity 96-4,700), far from any allocation worth deploying.

Every evaluation is saved at once to results/phase4_scan.json, so a rerun resumes where it stopped.
Usage: python sensitivity_scan.py [--only scan|pairs|allocations|near]
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch

from common import (DATA, ROOT, disable_power_throttling, keep_awake, load_model, read_weights, versions,
                    window_nll, write_json)
from quant import KINDS, MODULES, WIDTHS, conv1d, fake_quant

CALIB_TOKENS, WINDOW = 16 * 1024, 1024
PAIR_BITS, PAIRS_PER_STRATUM = 3, 10
ALLOCATION_BUDGETS, ALLOCATIONS_PER_BUDGET, MOVES = [3, 4], 10, 40
NEAR_MOVES = [2, 2, 2, 4, 4, 4, 6, 6, 6]  # one random near-uniform allocation per entry, per budget
SEED = 0
OUT = ROOT / "results" / "phase4_scan.json"

# Width changes of +1 and +2 code bits within the width set {2, 3, 4, 6, 8}.
UP = {1: {2: 3, 3: 4}, 2: {2: 4, 4: 6, 6: 8}}
DOWN = {d: {v: k for k, v in m.items()} for d, m in UP.items()}


def block_and_kind(index):
    return index // len(KINDS), index % len(KINDS)


def pair_strata(rng):
    """Stratified module pairs: (same block | adjacent blocks | blocks 4+ apart) x (same type | different)."""
    pairs = {}
    for name, distance_ok in [("same_block", lambda d: d == 0), ("adjacent", lambda d: d == 1),
                              ("distant", lambda d: d >= 4)]:
        for type_name, same_type in [("same_type", True), ("different_type", False)]:
            if name == "same_block" and same_type:
                continue  # a block has one module of each type
            candidates = [(a, b) for a in range(48) for b in range(a + 1, 48)
                          if distance_ok(abs(block_and_kind(a)[0] - block_and_kind(b)[0]))
                          and (block_and_kind(a)[1] == block_and_kind(b)[1]) == same_type]
            k = 2 * PAIRS_PER_STRATUM if name == "same_block" else PAIRS_PER_STRATUM
            pairs[f"{name}/{type_name}"] = rng.sample(candidates, k)
    return pairs


def random_allocation(rng, bits, moves=MOVES):
    """Start from uniform `bits` and make `moves` byte-neutral moves: raise one module and lower another
    module of the same type (so the same size) by the same number of code bits."""
    widths = [bits] * 48
    for _ in range(moves):
        delta, kind = rng.choice([1, 2]), rng.randrange(len(KINDS))
        same = [i for i in range(48) if block_and_kind(i)[1] == kind]
        up = [i for i in same if widths[i] in UP[delta]]
        down = [i for i in same if widths[i] in DOWN[delta]]
        if not up or not down:
            continue
        a, b = rng.choice(up), rng.choice(down)
        if a != b:
            widths[a], widths[b] = UP[delta][widths[a]], DOWN[delta][widths[b]]
    return widths


class Evaluator:
    def __init__(self):
        self.model = load_model()
        tokens = np.fromfile(DATA / "wikitext2_train.u16", dtype="<u2")[:CALIB_TOKENS].astype(np.int64)
        self.tokens = torch.from_numpy(tokens)
        # FP32 originals come from the exported file (bit-identical to the model, Phase 0 gate), memory-mapped:
        # keeping copies of all 48 weights would take another 340 MB of RAM.
        self.file = read_weights(DATA / "gpt2_124M.bin")
        self.seconds = []

    def original(self, i):
        return torch.from_numpy(np.array(self.file[MODULES[i] + ".weight"]))

    def perplexity(self, widths):
        """Calibration perplexity with module i quantized to widths[i] bits; restores FP32 afterwards."""
        with torch.inference_mode():
            for i, b in widths.items():
                conv1d(self.model, MODULES[i]).weight.copy_(fake_quant(self.original(i), b))
            t0 = time.perf_counter()
            nll = window_nll(self.model, self.tokens, WINDOW)
            self.seconds.append(time.perf_counter() - t0)
            for i in widths:
                conv1d(self.model, MODULES[i]).weight.copy_(self.original(i))
        return math.exp(sum(nll) / len(nll))


def save(state):
    state["seconds_per_evaluation"] = float(np.mean(state["_seconds"])) if state["_seconds"] else None
    write_json(OUT, {k: v for k, v in state.items() if not k.startswith("_")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["scan", "pairs", "allocations", "near"])
    args = parser.parse_args()
    print(f"power throttling disabled: {disable_power_throttling()}", flush=True)
    print(f"sleep blocked: {keep_awake()}", flush=True)
    torch.manual_seed(SEED)
    state = json.loads(OUT.read_text()) if OUT.exists() else {
        "calibration": {"tokens": f"wikitext2_train.u16[0:{CALIB_TOKENS}]", "window": WINDOW, "stride": WINDOW},
        "modules": MODULES, "widths": WIDTHS, "P0": None, "P": {}, "pairs": {}, "allocations": {}}
    state.setdefault("allocations_near", {})
    state["torch_threads"], state["versions"] = torch.get_num_threads(), versions()
    ev = Evaluator()
    state["_seconds"] = ev.seconds
    OUT.parent.mkdir(exist_ok=True)

    def run(label, widths):
        p = ev.perplexity(widths)
        print(f"{label}: perplexity {p:.4f} ({ev.seconds[-1]:.0f} s)", flush=True)
        return p

    if args.only in (None, "scan"):
        if state["P0"] is None:
            state["P0"] = run("FP32", {})
            save(state)
        for i in range(48):
            done = state["P"].setdefault(str(i), {})
            for b in WIDTHS:
                if str(b) not in done:
                    done[str(b)] = run(f"module {i:2d} {MODULES[i]} {b}-bit", {i: b})
                    save(state)

    rng = random.Random(SEED)
    pairs = pair_strata(rng)  # drawn first so the allocations below do not depend on --only
    if args.only in (None, "pairs"):
        for stratum, members in pairs.items():
            for a, b in members:
                key = f"{a},{b}"
                if key not in state["pairs"]:
                    state["pairs"][key] = {"stratum": stratum, "bits": PAIR_BITS,
                                           "P": run(f"pair {stratum} {a},{b}", {a: PAIR_BITS, b: PAIR_BITS})}
                    save(state)

    if args.only in (None, "allocations"):
        for bits in ALLOCATION_BUDGETS:
            for k in range(ALLOCATIONS_PER_BUDGET):
                widths = random_allocation(rng, bits)
                key = f"{bits}-bit #{k}"
                if key not in state["allocations"]:
                    state["allocations"][key] = {"widths": widths,
                                                 "P": run(f"allocation {key}", dict(enumerate(widths)))}
                    save(state)

    if args.only in (None, "near"):
        near_rng = random.Random(SEED + 1)  # its own stream, so the sets above stay as they were
        for bits in ALLOCATION_BUDGETS:
            plan = [("uniform", 0)] + [(f"#{k}", m) for k, m in enumerate(NEAR_MOVES)]
            for name, moves in plan:
                widths = random_allocation(near_rng, bits, moves)
                key = f"{bits}-bit {name}"
                if key not in state["allocations_near"]:
                    state["allocations_near"][key] = {"widths": widths, "moves": moves,
                                                      "P": run(f"near allocation {key} ({moves} moves)",
                                                               dict(enumerate(widths)))}
                    save(state)


if __name__ == "__main__":
    main()
