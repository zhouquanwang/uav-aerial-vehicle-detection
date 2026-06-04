"""Resolve YAML device strings to SAHI / PyTorch device ids."""

from __future__ import annotations

ConfiguredDevice = str | int | None


def resolve_device(configured: ConfiguredDevice) -> str:
    import torch

    if configured is None:
        raw = "auto"
    elif isinstance(configured, int):
        return str(configured) if configured >= 0 else "cpu"
    else:
        raw = str(configured).strip()
    key = raw.lower()

    if key in ("auto", "gpu", "cuda"):
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if key == "cpu":
        return "cpu"
    return raw


def device_status_line(resolved: str) -> str:
    import torch

    parts = [f"device: {resolved}"]
    if torch.cuda.is_available():
        idx = 0
        try:
            if resolved.isdigit():
                idx = int(resolved)
            elif resolved.startswith("cuda:"):
                idx = int(resolved.split(":")[1])
        except (ValueError, IndexError):
            idx = 0
        name = torch.cuda.get_device_name(idx)
        parts.append(f"CUDA {torch.version.cuda} | GPU: {name}")
    else:
        parts.append("CUDA unavailable (CPU)")
    return " | ".join(parts)
