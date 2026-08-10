"""NVIDIA RTX Video Super Resolution Pro node for ComfyUI.

Derived from Comfy-Org/Nvidia_RTX_Nodes_ComfyUI under Apache-2.0.
The public node id is intentionally preserved for workflow compatibility.
"""

from __future__ import annotations

import gc
import logging
from enum import Enum
from typing import TypedDict

import nvvfx
import torch
import torch.nn.functional as functional
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .vsr_pro import (
    VERIFIED_VSR_MAX_EDGE,
    assert_channel_integrity,
    plan_dimensions,
    print_dimensions_to_pixels,
)


LOGGER = logging.getLogger(__name__)


class UpscaleType(str, Enum):
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"
    PRINT_DIMENSIONS = "print size (mm + DPI)"


class RTXVideoSuperResolution(io.ComfyNode):
    """RTX VSR with verified large-output planning and corruption detection."""

    class UpscaleTypedDict(TypedDict):
        resize_type: UpscaleType
        scale: float
        width: int
        height: int
        width_mm: float
        height_mm: float
        dpi: int

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXVideoSuperResolution",
            display_name="RTX Video Super Resolution Pro",
            category="image/upscaling",
            description=(
                "GPU-accelerated NVIDIA RTX VSR Pro with automatic 16K protection. "
                "Large targets are enhanced at a validated intermediate size "
                "and then resized to the exact requested dimensions."
            ),
            search_aliases=["rtx", "nvidia", "upscale", "super resolution", "vsr", "pro 16k"],
            inputs=[
                io.Image.Input("images"),
                io.DynamicCombo.Input(
                    "resize_type",
                    tooltip="Choose to scale by a multiplier or to exact target dimensions.",
                    options=[
                        io.DynamicCombo.Option(
                            UpscaleType.SCALE_BY,
                            [
                                io.Float.Input(
                                    "scale",
                                    default=2.0,
                                    min=1.0,
                                    max=4.0,
                                    step=0.01,
                                    tooltip="Scale factor (for example, 4.0 turns 4096 px into 16384 px with the Pro hybrid path).",
                                ),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            UpscaleType.TARGET_DIMENSIONS,
                            [
                                io.Int.Input("width", default=1920, min=64, max=16384, step=8),
                                io.Int.Input("height", default=1080, min=64, max=16384, step=8),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            UpscaleType.PRINT_DIMENSIONS,
                            [
                                io.Float.Input(
                                    "width_mm",
                                    default=500.0,
                                    min=1.0,
                                    max=5000.0,
                                    step=1.0,
                                    tooltip="Final physical print width in millimetres.",
                                ),
                                io.Float.Input(
                                    "height_mm",
                                    default=500.0,
                                    min=1.0,
                                    max=5000.0,
                                    step=1.0,
                                    tooltip="Final physical print height in millimetres.",
                                ),
                                io.Int.Input(
                                    "dpi",
                                    default=300,
                                    min=72,
                                    max=1200,
                                    step=1,
                                    tooltip="Print resolution. Pixels are calculated as mm / 25.4 x DPI.",
                                ),
                            ],
                        ),
                    ],
                ),
                io.Combo.Input(
                    "quality",
                    options=["LOW", "MEDIUM", "HIGH", "ULTRA"],
                    default="ULTRA",
                ),
            ],
            outputs=[
                io.Image.Output("upscaled_images"),
                io.Float.Output(
                    "calculated_scale",
                    tooltip="Largest calculated output/input edge ratio.",
                ),
                io.Int.Output("output_width", tooltip="Final aligned pixel width."),
                io.Int.Output("output_height", tooltip="Final aligned pixel height."),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        resize_type: UpscaleTypedDict,
        quality: str,
    ) -> io.NodeOutput:
        if images.ndim != 4:
            raise ValueError(f"Expected ComfyUI IMAGE tensor [B,H,W,C], got shape {tuple(images.shape)}")

        image_count, input_height, input_width, channels = images.shape
        if channels != 3:
            raise ValueError(
                "RTX VSR requires exactly three RGB channels. "
                f"Received {channels}; flatten alpha before this node."
            )

        selected_type = resize_type["resize_type"]
        if selected_type == UpscaleType.SCALE_BY:
            scale = resize_type["scale"]
            requested_width = int(input_width * scale)
            requested_height = int(input_height * scale)
        elif selected_type == UpscaleType.TARGET_DIMENSIONS:
            requested_width = resize_type["width"]
            requested_height = resize_type["height"]
        elif selected_type == UpscaleType.PRINT_DIMENSIONS:
            requested_width, requested_height = print_dimensions_to_pixels(
                width_mm=resize_type["width_mm"],
                height_mm=resize_type["height_mm"],
                dpi=resize_type["dpi"],
            )
            LOGGER.info(
                "Print target %.1fx%.1f mm at %s DPI requires %sx%s px before alignment.",
                resize_type["width_mm"],
                resize_type["height_mm"],
                resize_type["dpi"],
                requested_width,
                requested_height,
            )
        else:
            raise ValueError(f"Unsupported resize type: {selected_type}")

        plan = plan_dimensions(
            input_width=input_width,
            input_height=input_height,
            requested_width=requested_width,
            requested_height=requested_height,
        )

        if plan.uses_hybrid_fallback:
            LOGGER.warning(
                "RTX VSR target %sx%s reaches the affected 16K boundary; "
                "using %sx%s VSR intermediate and bicubic final resize.",
                plan.output_width,
                plan.output_height,
                plan.vsr_width,
                plan.vsr_height,
            )

        quality_mapping = {
            "LOW": nvvfx.effects.QualityLevel.LOW,
            "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
            "HIGH": nvvfx.effects.QualityLevel.HIGH,
            "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
        }
        selected_quality = quality_mapping.get(quality, nvvfx.effects.QualityLevel.HIGH)

        # Return CPU tensors like ComfyUI's image loader. This avoids keeping the
        # multi-gigabyte 16K result in VRAM after the node has completed.
        output = torch.empty(
            (image_count, plan.output_height, plan.output_width, channels),
            device="cpu",
            dtype=images.dtype,
        )

        try:
            with nvvfx.VideoSuperRes(selected_quality) as super_resolution:
                super_resolution.output_width = plan.vsr_width
                super_resolution.output_height = plan.vsr_height
                super_resolution.load()

                for index in range(image_count):
                    input_frame = (
                        images[index]
                        .movedim(-1, 0)
                        .to(device="cuda", dtype=torch.float32)
                        .contiguous()
                    )

                    result = super_resolution.run(input_frame)
                    # The nvvfx API states that the returned DLPack capsule points
                    # at SDK-owned memory. Clone immediately before the next run.
                    vsr_frame = torch.from_dlpack(result.image).clone()
                    del result

                    assert_channel_integrity(input_frame, vsr_frame)

                    if plan.uses_hybrid_fallback:
                        vsr_frame = functional.interpolate(
                            vsr_frame.unsqueeze(0),
                            size=(plan.output_height, plan.output_width),
                            mode="bicubic",
                            align_corners=False,
                            antialias=True,
                        ).squeeze(0)

                    final_frame = vsr_frame.clamp_(0.0, 1.0).movedim(0, -1)
                    output[index].copy_(final_frame.to(device="cpu", dtype=images.dtype))
                    del input_frame, vsr_frame, final_frame
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        calculated_scale = max(
            plan.output_width / input_width,
            plan.output_height / input_height,
        )
        return io.NodeOutput(
            output,
            calculated_scale,
            plan.output_width,
            plan.output_height,
        )


class NVVFXVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [RTXVideoSuperResolution]


async def comfy_entrypoint() -> NVVFXVideoExtension:
    return NVVFXVideoExtension()


# Compatibility hint used by registry scanners.
if False:
    NODE_CLASS_MAPPINGS = {"RTXVideoSuperResolution": RTXVideoSuperResolution}


__all__ = ["RTXVideoSuperResolution", "comfy_entrypoint", "VERIFIED_VSR_MAX_EDGE"]
