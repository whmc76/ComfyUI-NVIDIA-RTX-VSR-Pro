"""Pure dimension planning and validation helpers for RTX VSR Pro."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import torch


ALIGNMENT = 8
MAX_OUTPUT_EDGE = 16384
# 15360x15360 is verified on the affected Windows/NVIDIA setup. The stock
# node produced a zeroed blue channel at exactly 16384x16384.
VERIFIED_VSR_MAX_EDGE = 15360


@dataclass(frozen=True)
class DimensionPlan:
    output_width: int
    output_height: int
    vsr_width: int
    vsr_height: int

    @property
    def uses_hybrid_fallback(self) -> bool:
        return (self.output_width, self.output_height) != (self.vsr_width, self.vsr_height)


def _align_nearest(value: int) -> int:
    return max(ALIGNMENT, round(value / ALIGNMENT) * ALIGNMENT)


def _align_down(value: float) -> int:
    return max(ALIGNMENT, floor(value / ALIGNMENT) * ALIGNMENT)


def plan_dimensions(
    *,
    input_width: int,
    input_height: int,
    requested_width: int,
    requested_height: int,
) -> DimensionPlan:
    """Plan exact output and a verified VSR intermediate, aligned to 8 px."""

    if min(input_width, input_height, requested_width, requested_height) <= 0:
        raise ValueError("Input and output dimensions must be positive")

    output_width = _align_nearest(requested_width)
    output_height = _align_nearest(requested_height)

    if max(output_width, output_height) > MAX_OUTPUT_EDGE:
        raise ValueError(
            "RTX VSR Pro supports a maximum output edge of "
            f"{MAX_OUTPUT_EDGE} px. Requested {output_width}x{output_height}; "
            "reduce the print size or DPI, or use a tiled workflow."
        )

    if output_width < input_width or output_height < input_height:
        raise ValueError(
            "RTX Video Super Resolution only supports upscaling. "
            f"Input is {input_width}x{input_height}, requested output is "
            f"{output_width}x{output_height}."
        )

    if max(output_width, output_height) <= VERIFIED_VSR_MAX_EDGE:
        return DimensionPlan(output_width, output_height, output_width, output_height)

    reduction = VERIFIED_VSR_MAX_EDGE / max(output_width, output_height)
    vsr_width = _align_down(output_width * reduction)
    vsr_height = _align_down(output_height * reduction)

    if vsr_width < input_width or vsr_height < input_height:
        raise ValueError(
            "The requested aspect ratio cannot be routed through the verified VSR "
            f"intermediate without downscaling ({vsr_width}x{vsr_height})."
        )

    return DimensionPlan(output_width, output_height, vsr_width, vsr_height)


def print_dimensions_to_pixels(*, width_mm: float, height_mm: float, dpi: int) -> tuple[int, int]:
    """Convert physical millimetres and DPI to the nearest pixel dimensions."""

    if width_mm <= 0 or height_mm <= 0 or dpi <= 0:
        raise ValueError("Print width, height, and DPI must be positive")
    width_px = round(width_mm / 25.4 * dpi)
    height_px = round(height_mm / 25.4 * dpi)
    return max(1, width_px), max(1, height_px)


def assert_channel_integrity(input_frame: torch.Tensor, output_frame: torch.Tensor) -> None:
    """Reject the catastrophic single-channel loss observed at the 16K boundary."""

    if input_frame.ndim != 3 or output_frame.ndim != 3:
        raise ValueError("Channel validation expects [C,H,W] tensors")
    if input_frame.shape[0] != 3 or output_frame.shape[0] != 3:
        raise ValueError("Channel validation expects RGB tensors")

    # Sample at most roughly 512x512 pixels per channel. The observed SDK
    # failure affects the complete channel, and sampling avoids allocating a
    # 16K-sized boolean tensor merely for validation.
    input_stride_y = max(1, input_frame.shape[1] // 512)
    input_stride_x = max(1, input_frame.shape[2] // 512)
    output_stride_y = max(1, output_frame.shape[1] // 512)
    output_stride_x = max(1, output_frame.shape[2] // 512)
    input_sample = input_frame[:, ::input_stride_y, ::input_stride_x]
    output_sample = output_frame[:, ::output_stride_y, ::output_stride_x]

    if not torch.isfinite(output_sample).all():
        raise RuntimeError("NVIDIA RTX VSR returned NaN or infinite pixel values")

    source_means = input_sample.float().mean(dim=(1, 2))
    output_means = output_sample.float().mean(dim=(1, 2))

    for channel, name in enumerate(("red", "green", "blue")):
        source_has_signal = source_means[channel] > 0.02
        output_collapsed = output_means[channel] < 0.003
        other_output_has_signal = torch.cat(
            (output_means[:channel], output_means[channel + 1 :])
        ).max() > 0.02
        if bool(source_has_signal and output_collapsed and other_output_has_signal):
            raise RuntimeError(
                "NVIDIA RTX VSR returned a corrupted image: "
                f"the {name} channel collapsed from mean "
                f"{source_means[channel].item():.4f} to "
                f"{output_means[channel].item():.4f}. No output was emitted."
            )
