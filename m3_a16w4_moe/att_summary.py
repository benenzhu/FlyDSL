#!/usr/bin/env python3
"""Summarize parsed rocprofv3 ATT wave traces for the decode GEMMs.

Input is a ui_output_* directory containing code.json and se*_wv*.json.
No GPU execution is involved. Durations are trace cycles, not kernel latency:
waves overlap, and a stall in one wave may be hidden by another wave's work.
For a persistent kernel, repeated executions of its first MFMA identify items.
"""

import argparse
import collections
import json
from pathlib import Path
import statistics


def distribution(values):
    if not values:
        return None
    return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(directory):
    directory = Path(directory)
    code_data = json.loads((directory / "code.json").read_text())
    assert code_data["header"].startswith("ISA, _, LineNumber"), code_data["header"]
    code = {row[2]: row[0] for row in code_data["code"]}
    waves = []
    front, compute, boundary, tail, total_per_item = [], [], [], [], []
    item_spans = []
    for path in sorted(directory.glob("se*_wv*.json")):
        data = json.loads(path.read_text())
        wave = data["wave"]
        inst = wave["instructions"]
        # Parsed instruction rows: [start_cycle, category, stall, duration, code_line].
        mma = [j for j, row in enumerate(inst) if "v_mfma" in code.get(row[4], "")]
        if not mma:
            continue  # bound-check-only CTAs
        first_line = inst[mma[0]][4]
        item_first = [j for j in mma if inst[j][4] == first_line]
        ends = item_first[1:] + [len(inst)]
        spans = []
        item_last = []
        for lo, hi in zip(item_first, ends):
            last = max(j for j in mma if lo <= j < hi)
            item_last.append(last)
            spans.append(inst[last][0] + inst[last][3] - inst[lo][0])
        nitems = len(item_first)
        front.append(inst[item_first[0]][0] - wave["begin"])
        compute.extend(spans)
        tail.append(wave["end"] - inst[item_last[-1]][0] - inst[item_last[-1]][3])
        total_per_item.append((wave["end"] - wave["begin"]) / nitems)
        for prev, nxt in zip(item_last, item_first[1:]):
            boundary.append(inst[nxt][0] - inst[prev][0] - inst[prev][3])
        for lo, hi in zip(item_first, item_first[1:]):
            item_spans.append(inst[hi][0] - inst[lo][0])
        waves.append({"file": path.name, "items": nitems, "mfmas": len(mma),
                      "cycles": wave["end"] - wave["begin"]})
    return {
        "directory": str(directory), "unit": "trace cycles",
        "waves": len(waves), "items_per_wave": dict(collections.Counter(w["items"] for w in waves)),
        "mfmas_per_item": dict(collections.Counter(w["mfmas"] / w["items"] for w in waves)),
        "first_item_prologue": distribution(front),
        "first_to_last_mfma": distribution(compute),
        "between_item_mfmas": distribution(boundary),
        "steady_item_first_mfma_interval": distribution(item_spans),
        "last_item_tail": distribution(tail),
        "wave_lifetime_per_item": distribution(total_per_item),
        "hotspots": "Use .claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py for stall classification and source attribution.",
        "caveat": "Per-wave intervals overlap across waves; do not sum them into whole-kernel time or infer HBM bandwidth.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    args = parser.parse_args()
    for directory in args.directories:
        print(json.dumps(summarize(directory), indent=2))
