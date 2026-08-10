# Changelog

## 0.4.0 - 2026-08-10

- Add a default-on **keep aspect ratio** switch to pixel-target and print-size
  modes. It fits the source inside the requested box without stretching or
  cropping; disabling it preserves exact width and height behavior.
- Treat missing switch values from older workflows as enabled.

## 0.3.0 - 2026-08-10

- Preserve the original node id for drop-in workflow compatibility.
- Add Pro hybrid processing for targets above `15360 px`, including exact
  `16384 x 16384` output.
- Detect catastrophic RGB channel collapse before emitting an image.
- Clone SDK-owned DLPack memory immediately and release VRAM after completion.
- Raise exact target controls to `16384 px`.
- Add a print-size mode that converts width/height in millimetres and DPI to
  pixels and reports the calculated scale and final aligned dimensions.
