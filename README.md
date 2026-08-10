# ComfyUI NVIDIA RTX VSR Safe

A workflow-compatible replacement for
[`Comfy-Org/Nvidia_RTX_Nodes_ComfyUI`](https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI)
that prevents the silent color-channel corruption observed at the exact
`16384 x 16384` output boundary.

The project keeps the original `RTXVideoSuperResolution` node id, so existing
workflows continue to load after the stock node is removed and this package is
installed.

## What is improved

- Preserves exact requested output dimensions up to `16384 x 16384`.
- Routes unsafe large targets through a verified `15360 px` RTX VSR
  intermediate, then performs an antialiased bicubic resize to the exact target.
- Detects catastrophic red, green, or blue channel collapse and stops instead
  of silently saving a corrupted image.
- Clones NVIDIA's SDK-owned DLPack result immediately, as required by the API.
- Processes frames sequentially and returns the completed image on CPU to
  release multi-gigabyte 16K allocations from VRAM.
- Expands exact target-dimension controls from `8192` to `16384`.

## Why the fallback exists

On the reproduced Windows setup, every `4096 x 4096 -> 16384 x 16384` run from
the stock node returned an image whose blue-channel mean was approximately
zero. Runs at `3840 x 3840 -> 15360 x 15360` preserved all three channels.
Red plus green without blue appears yellow, which explains the visible failure.

The safe node still applies NVIDIA RTX VSR to nearly the entire requested
resolution. Only the final `6.67%` edge increase from `15360` to `16384` uses
standard antialiased interpolation.

## Installation

Remove or disable the stock node first. Both packages intentionally register
the same node id and must not be enabled together.

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/whmc76/ComfyUI-NVIDIA-RTX-VSR-Safe.git
..\..\python_embeded\python.exe -m pip install -r ComfyUI-NVIDIA-RTX-VSR-Safe\requirements.txt
```

Restart ComfyUI. Existing workflows should display **RTX Video Super
Resolution (Safe)** in place of the stock node.

## Compatibility

- Windows or Linux with an NVIDIA RTX GPU supported by `nvidia-vfx`
- Current ComfyUI extension API (`comfy_api.latest`)
- `nvidia-vfx 0.1.0.1` at the time of the first release

The NVIDIA Video Effects SDK remains proprietary and is installed separately
through the `nvidia-vfx` Python package. This repository contains no NVIDIA
model weights or SDK binaries.

## Development check

Run the narrow safety checks with the Python environment that provides PyTorch:

```powershell
python -m unittest discover -s tests -v
```

## License and attribution

Apache License 2.0. Derived from the Apache-2.0 licensed
`Comfy-Org/Nvidia_RTX_Nodes_ComfyUI` project. See [NOTICE](NOTICE).
