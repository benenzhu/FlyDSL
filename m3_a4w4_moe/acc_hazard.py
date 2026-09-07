# For an AMDGPU .s: for every v_accvgpr_read of a[k], find the most recent v_mfma writing a range that
# contains k and count the real instructions between them (a proxy for wait states, excluding s_nop's
# own count which is added as its immediate+1). Report the minimum and the count below 20.
import re, sys
lines = open(sys.argv[1]).read().split("\n")
ins = []
for i, l in enumerate(lines):
    t = l.strip()
    if not t or t.startswith((";", ".", "//")) or t.endswith(":"): continue
    ins.append((i, t))
last_write = {}   # agpr index -> position in ins
worst = []
for pos, (ln, t) in enumerate(ins):
    m = re.match(r"v_mfma\S*\s+a\[(\d+):(\d+)\]", t)
    if m:
        for k in range(int(m.group(1)), int(m.group(2)) + 1): last_write[k] = pos
        continue
    m = re.match(r"v_accvgpr_read_b32\s+v\d+,\s+a(\d+)", t)
    if m:
        k = int(m.group(1))
        if k in last_write:
            w = last_write[k]
            ws = 0
            for p in range(w + 1, pos):
                tt = ins[p][1]
                mm = re.match(r"s_nop\s+(\d+)", tt)
                ws += int(mm.group(1)) + 1 if mm else 1
            worst.append((ws, ln, ins[w][0]))
worst.sort()
print(f"{sys.argv[1]}: {len(worst)} accvgpr reads with a preceding mfma write; min wait-states {worst[0][0] if worst else None}; below 20: {sum(1 for w in worst if w[0] < 20)}; below 40: {sum(1 for w in worst if w[0] < 40)}")
for ws, ln, wl in worst[:6]:
    print(f"   {ws:4d} wait states: read at line {ln+1}, mfma write at line {wl+1}")
