#!/usr/bin/env python3
"""Generate a compact hot-loop instruction table from a (pure) AMDGCN ISA file.

Reduces the steady-state main loop to the three memory/compute ops that drive
GEMM scheduling, one per line, using single-letter tags:

    M = v_mfma_*        (matrix multiply-accumulate)
    r = ds_read_*       (LDS -> register, S2R)
    L = buffer_load_*   (global -> LDS/VGPR prefetch, G2S)
    B = s_barrier       (half / loop boundary)

All other instructions (address calc, waitcnt, salu, valu) are dropped so the
M/r/L interleaving and the barrier-delimited halves are visible at a glance.

Usage:
    hotloop_table.py <isa.s> [--loop-label .LBB0_1] [--out table.txt]
                     [--compact]   # also emit a one-char-per-op stream line

If --loop-label is omitted, the script auto-detects the main loop as the body
between the most-branched-to backward label and its s_cbranch.
"""
import argparse
import re
import sys
from collections import Counter

TAGS = [
    ("M", re.compile(r"^\s*v_mfma")),
    ("r", re.compile(r"^\s*ds_read")),
    ("L", re.compile(r"^\s*buffer_load")),
    ("B", re.compile(r"^\s*s_barrier")),
]


def classify(line: str):
    for tag, pat in TAGS:
        if pat.match(line):
            return tag
    return None


def find_main_loop(lines):
    # Pick the backward branch target that a s_cbranch jumps to, preferring the
    # one whose body contains the most v_mfma (the steady-state K loop).
    labels = {l.strip().rstrip(":"): i for i, l in enumerate(lines) if re.match(r"^\.\w", l.strip()) and l.strip().endswith(":")}
    best = None
    for i, l in enumerate(lines):
        m = re.search(r"s_cbranch\w*\s+(\.\S+)", l)
        if not m:
            continue
        tgt = m.group(1)
        if tgt in labels and labels[tgt] < i:
            start = labels[tgt]
            body = lines[start : i + 1]
            nmfma = sum(1 for b in body if b.strip().startswith("v_mfma"))
            if best is None or nmfma > best[0]:
                best = (nmfma, tgt, start, i)
    if best is None:
        raise SystemExit("could not auto-detect a backward loop; pass --loop-label")
    return best[1], best[2], best[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("isa")
    ap.add_argument("--loop-label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compact", action="store_true")
    args = ap.parse_args()

    lines = open(args.isa).read().splitlines()

    if args.loop_label:
        labels = {l.strip().rstrip(":"): i for i, l in enumerate(lines) if l.strip().endswith(":")}
        start = labels[args.loop_label]
        end = next(i for i, l in enumerate(lines) if i > start and "s_cbranch" in l and args.loop_label in l)
        label = args.loop_label
    else:
        label, start, end = find_main_loop(lines)

    body = lines[start : end + 1]
    seq = []  # (tag, operands-summary)
    counts = Counter()
    for l in body:
        tag = classify(l)
        if tag is None:
            continue
        counts[tag] += 1
        s = l.strip()
        # keep just mnemonic + first dst operand for context
        parts = s.split(None, 1)
        mnem = parts[0]
        rest = parts[1].split(";")[0].strip() if len(parts) > 1 else ""
        dst = rest.split(",")[0].strip() if rest else ""
        seq.append((tag, mnem, dst))

    out = []
    out.append(f"Hot-loop table for main loop {label}  (ISA: {args.isa})")
    out.append("Tags: M=v_mfma  r=ds_read  L=buffer_load  B=s_barrier")
    out.append(
        f"Per loop: M={counts['M']} r={counts['r']} L={counts['L']} B={counts['B']}"
    )
    out.append("=" * 60)
    half = 0
    for tag, mnem, dst in seq:
        if tag == "B":
            out.append(f"B  -------- barrier (end of half {half}) --------")
            half += 1
        else:
            out.append(f"{tag}  {mnem:28s} {dst}")

    if args.compact:
        stream = "".join(t for t, _, _ in seq)
        out.append("=" * 60)
        out.append("stream: " + stream)

    text = "\n".join(out) + "\n"
    if args.out:
        open(args.out, "w").write(text)
        print(f"wrote {args.out}  ({len(seq)} ops)")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
