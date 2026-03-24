"""Optional PyTorch profiler helpers (Chrome trace export)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch


def profile_training_steps(
    device: torch.device,
    train_step: Callable[[], None],
    output_dir: str | Path,
    warmup: int = 2,
    active: int = 8,
) -> Path:
    """
    Run ``train_step()`` repeatedly under torch.profiler and export traces for TensorBoard/Chrome.

    ``train_step`` should perform one training iteration (forward + backward + optimizer step).
    Returns the directory containing trace files (tensorboard_trace_handler output).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    schedule = torch.profiler.schedule(wait=0, warmup=warmup, active=active, repeat=1)
    total_steps = warmup + active + 1

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(total_steps):
            train_step()
            prof.step()

    summary_path = out / "profiler_summary.txt"
    summary_path.write_text(
        prof.key_averages().table(
            sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total",
            row_limit=40,
        ),
        encoding="utf-8",
    )
    return out
