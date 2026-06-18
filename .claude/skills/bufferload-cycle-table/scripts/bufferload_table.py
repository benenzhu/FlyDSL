#!/usr/bin/env python3
"""Per-tile buffer_load cycle/stall table from ATT, anchored at the tile barrier.

For one steady-state K tile, list every buffer_load as:
    gap(stall)
where gap = cycles since the previous buffer_load (the first one's gap is its
offset from the tile-start barrier), and stall = ATT stall cycles on that load.
A trailing `end-> gap(end)` row gives the cycles from the last load to the tile
end barrier. Compares two kernels side by side.

Why: if two GEMMs move the same bytes/tile but one stalls far more than the
MFMA-width difference can explain, the faster one's DMAs are completing faster /
the VMEM queue turns over faster -- not just hidden behind longer MFMAs. The
per-load gap+stall view (anchored at the tile barrier) exposes that. Note that a
LARGER gap with MORE stall rules out "spacing too tight"; it points at VMEM queue
back-pressure (DMA retire rate), which correlates with occupancy and dtype.

Usage:
    bufferload_table.py <ui_dir_A> <ui_dir_B> [--labels fp8,bf16]
                        [--bar-start 40] [--tiles 2] [--wave se0_sm0_sl0_wv0.json]

`--tiles N` = how many barrier intervals make one K tile (RP1 bf16 = 2; check the
kernel). Pick `--bar-start` in the steady state (e.g. 40).
"""
import argparse
import json
import os


def load_loads(ui_dir, bar_start, tiles, wave):
    j = json.load(open(os.path.join(ui_dir, wave)))
    insts = j["wave"]["instructions"]
    code = json.load(open(os.path.join(ui_dir, "code.json")))["code"]
    idmap = {c[2]: c[0] for c in code}  # code line index -> asm text
    # inst tuple = [cycle, type, stall, dur, code_id]
    bar = [t[0] for t in insts if "s_barrier" in str(idmap.get(t[4], ""))]
    b0 = bar[bar_start]
    b_end = bar[bar_start + tiles]
    out = []
    for t in insts:
        if b0 <= t[0] < b_end:
            asm = str(idmap.get(t[4], "")).strip()
            if asm.startswith("buffer_load"):
                out.append((t[0] - b0, t[2]))  # (offset_from_tile_start, stall)
    return out, b_end - b0


def to_gaps(loads, tilelen):
    rows = []
    prev = 0
    for off, st in loads:
        rows.append((off - prev, st))
        prev = off
    rows.append((tilelen - prev, "end"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--labels", default="A,B")
    ap.add_argument("--bar-start", type=int, default=40)
    ap.add_argument("--tiles", type=int, default=2)
    ap.add_argument("--wave", default="se0_sm0_sl0_wv0.json")
    args = ap.parse_args()
    la, lb = args.labels.split(",")

    a, al = load_loads(args.dir_a, args.bar_start, args.tiles, args.wave)
    b, bl = load_loads(args.dir_b, args.bar_start, args.tiles, args.wave)
    ga, gb = to_gaps(a, al), to_gaps(b, bl)

    n = max(len(ga), len(gb))
    print(f"{'':>5} {la+' gap(stall)':>16}  {lb+' gap(stall)':>16}   (gap=cyc since prev L)")
    print("-" * 56)
    for i in range(n):
        ca = f"{ga[i][0]}({ga[i][1]})" if i < len(ga) else ""
        cb = f"{gb[i][0]}({gb[i][1]})" if i < len(gb) else ""
        tag = "end->" if i == n - 1 else f"L{i:>2} "
        print(f"{tag:>5} {ca:>16}  {cb:>16}")
    print("-" * 56)
    print(f"{'sum':>5} {'stall='+str(sum(s for _,s in a)):>16}  {'stall='+str(sum(s for _,s in b)):>16}")
    print(f"{'len':>5} {al:>16}  {bl:>16}")


if __name__ == "__main__":
    main()
