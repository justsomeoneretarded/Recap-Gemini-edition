# Reference video measurements

Computed on the supplied release files; no source-film comparison or subjective audio audition.

| ID | Duration (s) | Changes >.12 | Changes >.25 | Median interval >.25 (s) | LUFS | LRA (LU) | True peak (dBFS) |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | 136.60 | 68 | 50 | 2.13 | -14.0 | 1.9 | -5.5 |
| V2 | 178.21 | 90 | 43 | 2.17 | -14.0 | 1.0 | -5.1 |
| V3 | 159.31 | 103 | 69 | 1.48 | -7.7 | 1.7 | 1.2 |
| V4 | 148.74 | 85 | 63 | 2.05 | -8.1 | 2.0 | 0.5 |
| V5 | 135.69 | 126 | 49 | 2.13 | -7.7 | 1.1 | 0.5 |
| V6 | 179.53 | 68 | 25 | 2.54 | -8.5 | 2.4 | 0.8 |
| V7 | 153.85 | 55 | 24 | 2.87 | -8.6 | 1.2 | 0.4 |
| V8 | 169.43 | 120 | 81 | 1.48 | -7.8 | 1.0 | 0.7 |

## Method and interpretation

FFmpeg decodes every video frame with one thread, downsizes to 270×270 and records scene-change scores above 0.12. The 0.25 subset is also reported. These are detector candidates, not verified shot counts. Threshold sensitivity is retained instead of presenting a single count as truth.

EBU R128 integrated loudness, loudness range and true peak describe the complete stereo mix. Silence detection uses −35 dB for at least 0.2 seconds. A continuous score does not mean continuous narration: background music may fill speech pauses.

All supplied files are square 1080×1080 H.264 streams at a container rate of 60 fps. This does not prove that the film source is 60 fps or that every frame is unique.

Per-video JSON includes all candidate timestamps and scores, command arguments, full-mix silence intervals and limitations.
