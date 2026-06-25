#!/usr/bin/env python3
"""Summarize ATT UI hot-loop cycles from a dispatch directory.

The script reads rocprofv3 ATT UI output directories containing code.json and
se*_sm*_sl*_wv*.json files. It measures either a barrier-to-barrier window or
an explicit cycle range for one wave.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path


def load_code(ui_dir: Path) -> dict[int, str]:
    code_path = ui_dir / "code.json"
    with code_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    code_by_id: dict[int, str] = {}
    for entry in payload["code"]:
        # Header is: ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle
        isa = entry[0]
        code_id = int(entry[2])
        code_by_id[code_id] = isa
    return code_by_id


def wave_files(ui_dir: Path, pattern: str) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(ui_dir / pattern)))


def choose_wave(ui_dir: Path, pattern: str, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else ui_dir / p
    candidates = wave_files(ui_dir, pattern)
    if not candidates:
        raise SystemExit(f"no wave json files matching {pattern!r} under {ui_dir}")

    def num_insts(path: Path) -> int:
        with path.open("r", encoding="utf-8") as f:
            return int(json.load(f).get("num_insts", 0))

    return max(candidates, key=num_insts)


def load_wave(path: Path) -> list[tuple[int, int]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    out: list[tuple[int, int]] = []
    for inst in payload["wave"]["instructions"]:
        cycle = int(inst[0])
        code_id = int(inst[-1])
        out.append((cycle, code_id))
    return out


def parse_range(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", text)
    if not m:
        raise argparse.ArgumentTypeError("expected START:END")
    start, end = int(m.group(1)), int(m.group(2))
    if end <= start:
        raise argparse.ArgumentTypeError("END must be greater than START")
    return start, end


def classify(isa: str) -> str:
    if "v_mfma" in isa:
        return "v_mfma"
    if "ds_read" in isa:
        return "ds_read"
    if "ds_write" in isa:
        return "ds_write"
    if "buffer_load" in isa:
        if " lds" in isa:
            return "buffer_load_lds"
        return "buffer_load"
    if "buffer_store" in isa:
        return "buffer_store"
    if isa.startswith("s_waitcnt"):
        return "s_waitcnt"
    if isa.startswith("s_barrier"):
        return "s_barrier"
    if isa.startswith("s_"):
        return "salu"
    if isa.startswith("v_"):
        return "valu"
    return "other"


def barrier_events(instructions: list[tuple[int, int]], code_by_id: dict[int, str]) -> list[tuple[int, int]]:
    events = []
    for cycle, code_id in instructions:
        if "s_barrier" in code_by_id.get(code_id, ""):
            events.append((cycle, code_id))
    return events


def summarize_window(
    instructions: list[tuple[int, int]],
    code_by_id: dict[int, str],
    start_cycle: int,
    end_cycle: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for cycle, code_id in instructions:
        if start_cycle <= cycle < end_cycle:
            counts[classify(code_by_id.get(code_id, ""))] += 1
    return counts


def print_barriers(events: list[tuple[int, int]], code_by_id: dict[int, str]) -> None:
    print(f"barriers={len(events)}")
    for i, (cycle, code_id) in enumerate(events):
        print(f"{i:4d} cycle={cycle} code_id={code_id} isa={code_by_id.get(code_id, '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ui_dir", type=Path, help="ATT ui_output_agent_*_dispatch_* directory")
    parser.add_argument("--wave-file", help="Wave JSON filename or absolute path")
    parser.add_argument("--wave-glob", default="se*_sm*_sl*_wv*.json", help="Wave JSON glob under ui_dir")
    parser.add_argument("--list-barriers", action="store_true", help="List s_barrier events and exit")
    parser.add_argument("--barrier-window", type=parse_range, help="Barrier ordinal window START:END")
    parser.add_argument("--cycle-range", type=parse_range, help="Explicit cycle window START:END")
    parser.add_argument("--auto-middle", type=int, help="Use N middle barrier intervals")
    parser.add_argument("--per-interval", action="store_true", help="Print cycles for each barrier interval")
    args = parser.parse_args()

    ui_dir = args.ui_dir.resolve()
    code_by_id = load_code(ui_dir)
    wave_path = choose_wave(ui_dir, args.wave_glob, args.wave_file).resolve()
    instructions = load_wave(wave_path)
    barriers = barrier_events(instructions, code_by_id)

    print(f"ui_dir={ui_dir}")
    print(f"wave={wave_path.name}")

    if args.list_barriers:
        print_barriers(barriers, code_by_id)
        return

    window_label = "cycle_range"
    loops = 1
    if args.cycle_range:
        start_cycle, end_cycle = args.cycle_range
    else:
        if len(barriers) < 2:
            raise SystemExit("need at least two s_barrier events or pass --cycle-range")
        if args.barrier_window:
            b_start, b_end = args.barrier_window
        else:
            n = args.auto_middle or min(20, len(barriers) - 1)
            if n <= 0:
                raise SystemExit("--auto-middle must be positive")
            b_start = max(0, (len(barriers) - 1 - n) // 2)
            b_end = b_start + n
        if b_start < 0 or b_end >= len(barriers) or b_end <= b_start:
            raise SystemExit(f"invalid barrier window {b_start}:{b_end}; have {len(barriers)} barriers")
        start_cycle = barriers[b_start][0]
        end_cycle = barriers[b_end][0]
        loops = b_end - b_start
        window_label = f"barrier_window={b_start}:{b_end}"

    counts = summarize_window(instructions, code_by_id, start_cycle, end_cycle)
    total_cycles = end_cycle - start_cycle
    mfma_count = counts["v_mfma"]

    print(window_label)
    print(f"cycle_start={start_cycle}")
    print(f"cycle_end={end_cycle}")
    print(f"total_cycles={total_cycles}")
    print(f"loops={loops}")
    print(f"cycles_per_loop={total_cycles / loops:.3f}")
    print(f"mfma_count={mfma_count}")
    if mfma_count:
        print(f"cycles_per_mfma16={total_cycles / (mfma_count * 16):.6f}")

    print("instruction_mix:")
    for key, value in counts.most_common():
        print(f"  {key}: {value}")

    if args.per_interval and not args.cycle_range:
        b_start, b_end = map(int, window_label.split("=")[1].split(":"))
        print("interval_cycles:")
        for i in range(b_start, b_end):
            c0 = barriers[i][0]
            c1 = barriers[i + 1][0]
            print(f"  {i:4d}->{i + 1:<4d} {c1 - c0}")


if __name__ == "__main__":
    main()
