# Changelog

## 0.2.0 - 2026-08-10

- Preserve the original node id for drop-in workflow compatibility.
- Add safe hybrid processing for targets above `15360 px`, including exact
  `16384 x 16384` output.
- Detect catastrophic RGB channel collapse before emitting an image.
- Clone SDK-owned DLPack memory immediately and release VRAM after completion.
- Raise exact target controls to `16384 px`.
