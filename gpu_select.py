import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuUsage:
    index: int
    total_used_bytes: int
    process_used_bytes: int
    gpu_busy_percent: int


def _parse_int(value: object) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _rocm_smi_json(*args: str) -> dict:
    proc = subprocess.run(
        ["rocm-smi", *args, "--json"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "rocm-smi failed")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to parse rocm-smi JSON: {exc}") from exc


def _rocm_smi_text(*args: str) -> str:
    proc = subprocess.run(
        ["rocm-smi", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "rocm-smi failed")
    return proc.stdout


def _card_total_used_bytes() -> dict[int, int]:
    data = _rocm_smi_json("--showmeminfo", "vram")
    result = {}
    for card, fields in data.items():
        match = re.fullmatch(r"card(\d+)", str(card))
        if not match:
            continue
        result[int(match.group(1))] = _parse_int(fields.get("VRAM Total Used Memory (B)"))
    return result


def _card_gpu_busy_percent() -> dict[int, int]:
    data = _rocm_smi_json("--showuse")
    result = {}
    for card, fields in data.items():
        match = re.fullmatch(r"card(\d+)", str(card))
        if not match:
            continue
        result[int(match.group(1))] = _parse_int(fields.get("GPU use (%)"))
    return result


def _pid_gpu_map(text: str) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    current_pid: int | None = None
    for line in text.splitlines():
        match = re.match(r"^PID\s+(\d+)\s+is using", line.strip())
        if match:
            current_pid = int(match.group(1))
            result.setdefault(current_pid, [])
            continue
        if current_pid is None:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"\d+(?:\s+\d+)*", stripped):
            result[current_pid].extend(int(item) for item in stripped.split())
            current_pid = None
    return result


def _pid_vram_map(text: str) -> dict[int, int]:
    result: dict[int, int] = {}
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("PID") and "VRAM USED" in stripped:
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped or stripped.startswith("="):
            continue
        parts = stripped.split()
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        result[int(parts[0])] = _parse_int(parts[3])
    return result


def _card_process_used_bytes(cards: set[int]) -> dict[int, int]:
    text = _rocm_smi_text("--showpidgpus", "--showpids")
    pid_to_gpus = _pid_gpu_map(text)
    pid_to_vram = _pid_vram_map(text)
    result = {card: 0 for card in cards}
    for pid, gpus in pid_to_gpus.items():
        vram = pid_to_vram.get(pid, 0)
        for gpu in gpus:
            if gpu in result:
                result[gpu] += vram
    return result


def get_gpu_usage() -> list[GpuUsage]:
    total_used = _card_total_used_bytes()
    if not total_used:
        raise RuntimeError("rocm-smi reported no GPU cards")
    gpu_busy = _card_gpu_busy_percent()
    process_used = _card_process_used_bytes(set(total_used))
    return [
        GpuUsage(
            index=index,
            total_used_bytes=total,
            process_used_bytes=process_used[index],
            gpu_busy_percent=gpu_busy.get(index, 0),
        )
        for index, total in sorted(total_used.items())
    ]


def select_empty_gpu(
    max_process_used_bytes: int = 1024,
    max_gpu_busy_percent: int = 0,
) -> GpuUsage:
    usages = get_gpu_usage()
    idle = [
        gpu
        for gpu in usages
        if gpu.process_used_bytes < max_process_used_bytes
        and gpu.gpu_busy_percent <= max_gpu_busy_percent
    ]
    if not idle:
        summary = ", ".join(
            f"gpu{gpu.index}:proc={gpu.process_used_bytes}B,"
            f"busy={gpu.gpu_busy_percent}%,total={gpu.total_used_bytes}B"
            for gpu in usages
        )
        raise RuntimeError(
            f"no GPU has process VRAM < {max_process_used_bytes} bytes "
            f"and GPU use <= {max_gpu_busy_percent}%; {summary}"
        )
    return min(
        idle,
        key=lambda gpu: (
            gpu.process_used_bytes,
            gpu.gpu_busy_percent,
            gpu.total_used_bytes,
            gpu.index,
        ),
    )


def bind_empty_gpu_for_torch(script_name: str, max_process_used_bytes: int = 1024) -> int | None:
    if os.environ.get("FLYDSL_AUTO_GPU", "1").lower() in {"0", "false", "no"}:
        return None
    already_bound = os.environ.get("FLYDSL_GPU_BOUND_FOR_TORCH")
    if already_bound is not None:
        try:
            return int(already_bound)
        except ValueError:
            return None

    try:
        selected = select_empty_gpu(max_process_used_bytes=max_process_used_bytes)
    except Exception as exc:
        print(f"[{script_name}] GPU auto-select failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    gpu = str(selected.index)
    os.environ["HIP_VISIBLE_DEVICES"] = gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ["FLYDSL_GPU_BOUND_FOR_TORCH"] = gpu
    print(
        f"[{script_name}] Auto-selected physical ROCm GPU/card {gpu} "
        f"(torch will see it as cuda:0; "
        f"process_vram={selected.process_used_bytes}B, "
        f"busy={selected.gpu_busy_percent}%, "
        f"total_used={selected.total_used_bytes}B)",
        flush=True,
    )
    return selected.index
