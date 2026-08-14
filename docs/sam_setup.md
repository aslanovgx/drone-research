## Performance Note

Tiling strategy: ~6-7 seconds per high-resolution image (9504×6336) on
RTX 3050 Ti. This is acceptable for offline processing of static
training images, but not practical for frame-by-frame video processing
at scale (e.g. 1800 frames/minute at 30fps would take ~3+ hours).

For video testing, recommended approach: frame sampling (process every
Nth frame) combined with resize strategy, or SAM2's video predictor
with memory-based tracking instead of per-frame automatic generation.
This needs confirmation from the project supervisor regarding video
processing speed requirements.