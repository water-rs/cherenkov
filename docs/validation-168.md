# External frame cost (#168)

Path `e` imports the frame in place. Path `c` copies the planes and converts them. Each cell is 240 measured frames after 30 warmup frames, paced at 120 Hz. The order inside a cell is ABAB (`e c e c`) and then the reversed BABA (`c e c e`).

GPU time for a frame is `gpu_seconds` plus `handoff_seconds`, with a missing stamp read as zero. That is the report's `total_seconds`. Host time for a frame is `encode_seconds` plus `submit_seconds`. Percentiles are nearest-rank, index `ceil(n·p) − 1`, the same function as `bench/src/report.rs`. The path table pools the four runs of a cell and path (960 frames). Each run's GPU percentiles match that run's stored `total_seconds`.

The difference table is a separate statistic. Position `i` pairs the ABAB run with the BABA run at that index, and the difference is path `e`'s run-level percentile minus path `c`'s. Four pairs. Each interval is 10000 bootstrap means of those four differences, drawn with replacement by `random.Random(168)` (`randrange(4)`), the generator restarted per cell and shared across the four metrics of that cell. The 95% bounds are the nearest-rank 0.025 and 0.975 of those means (indexes 249 and 9749).

Steady memory is the median of the four runs' steady snapshots. With four readings the median is the mean of the two central values. Engine GPU bytes were the same on all four runs of a path, so that median is the reading.

Times are milliseconds rounded to 0.001. The raw reports stay on the machines that measured them.

## iPhone 16 Pro

Metal, Apple A18 Pro, target `Rgba16Float`. Measured at `31ee6dd0d08b8b37ac363c3038ec6e3d64659762`, from 6:10:45 PM to 6:16:16 PM EDT on 2 October 2026. The installed host is CDHash `4405ad75b95aef805fbf1fea4d49deb8daf722f8`, executable sha256 `c592b474ef26a0c96f2872101cbb68572d7a92dd10e2eb7478e87e44add2d73c`. iOS drops `--energy` and `--cpu`. The device lock wait for this matrix was 0 s. `status.json` names these 32 runs. Fifteen other `run-*` directories in the out dir belong to an earlier driver and are not included.

`ProcessInfo.thermalState` was nominal on 29 runs and fair on the last three (all inside the nominal/fair gate).

| Cell | Path | GPU p50 | GPU p99 | Host p50 | Host p99 |
|---|---|---:|---:|---:|---:|
| 1080p SDR | e | 1.216 | 1.440 | 1.833 | 3.735 |
| 1080p SDR | c | 6.060 | 7.850 | 2.940 | 4.953 |
| 1080p PQ | e | 1.418 | 1.581 | 2.124 | 3.671 |
| 1080p PQ | c | 8.376 | 9.793 | 4.022 | 5.715 |
| 4K SDR | e | 4.585 | 4.792 | 2.750 | 5.546 |
| 4K SDR | c | 9.345 | 12.477 | 4.617 | 7.032 |
| 4K PQ | e | 5.225 | 5.402 | 3.454 | 7.440 |
| 4K PQ | c | 9.225 | 12.196 | 3.237 | 5.402 |

e − c, bootstrap mean and 95% interval:

| Cell | GPU p50 | GPU p99 | Host p50 | Host p99 |
|---|---|---|---|---|
| 1080p SDR | −4.810 [−5.017, −4.603] | −6.421 [−7.013, −5.828] | −1.123 [−1.316, −0.987] | −1.569 [−3.245, −0.526] |
| 1080p PQ | −6.935 [−7.090, −6.841] | −8.240 [−8.595, −7.884] | −1.893 [−2.056, −1.778] | −2.055 [−2.462, −1.393] |
| 4K SDR | −4.770 [−4.911, −4.629] | −7.573 [−8.313, −6.928] | −1.868 [−2.000, −1.707] | −1.710 [−3.139, −0.602] |
| 4K PQ | −4.029 [−4.073, −3.985] | −6.542 [−6.912, −6.142] | 0.220 [0.096, 0.345] | 2.637 [1.281, 4.207] |

On 4K PQ the host p50 difference is positive on every pair: path `e` spent more host time than path `c`. Three of those four pairs cross the thermal boundary (one run nominal, the other fair).

Steady footprint. The wgpu allocator report is unavailable: "wgpu backend Metal returns no allocator report".

| Cell | Path | phys_footprint bytes | engine GPU bytes |
|---|---|---:|---:|
| 1080p SDR | e | 109766400 | 16598208 |
| 1080p SDR | c | 140617496 | 33186816 |
| 1080p PQ | e | 119080728 | 16598208 |
| 1080p PQ | c | 155772720 | 33186816 |
| 4K SDR | e | 199051032 | 66364608 |
| 4K SDR | c | 301066032 | 132719616 |
| 4K PQ | e | 236906312 | 66364608 |
| 4K PQ | c | 363079532 | 132719616 |

| Run | Cell | Order | Rep | Path | Thermal |
|---|---|---|---:|---|---|
| 62e0dabc2656 | 1080p SDR | abab | 0 | e | nominal |
| 8e58ca1e6cc4 | 1080p SDR | abab | 1 | c | nominal |
| 88d1a4972954 | 1080p SDR | abab | 2 | e | nominal |
| 94b31d31b435 | 1080p SDR | abab | 3 | c | nominal |
| f9a9cf5258a8 | 1080p SDR | baba | 0 | c | nominal |
| e8aa3ffc2437 | 1080p SDR | baba | 1 | e | nominal |
| 1985282bb9b3 | 1080p SDR | baba | 2 | c | nominal |
| 7dd8d4e9de04 | 1080p SDR | baba | 3 | e | nominal |
| b54870b83d8a | 1080p PQ | abab | 0 | e | nominal |
| bd1c163aef82 | 1080p PQ | abab | 1 | c | nominal |
| 9701384ab998 | 1080p PQ | abab | 2 | e | nominal |
| f0a63d2e3694 | 1080p PQ | abab | 3 | c | nominal |
| eab89a82db5d | 1080p PQ | baba | 0 | c | nominal |
| fc9c3aa74508 | 1080p PQ | baba | 1 | e | nominal |
| 7408b72ffeba | 1080p PQ | baba | 2 | c | nominal |
| ad09dd1fac50 | 1080p PQ | baba | 3 | e | nominal |
| 099c9d0c4712 | 4K SDR | abab | 0 | e | nominal |
| fbe4459f0606 | 4K SDR | abab | 1 | c | nominal |
| 7a78accf4dd6 | 4K SDR | abab | 2 | e | nominal |
| f0260064c700 | 4K SDR | abab | 3 | c | nominal |
| 15d9b6709014 | 4K SDR | baba | 0 | c | nominal |
| d85678b5a02f | 4K SDR | baba | 1 | e | nominal |
| 76a377bf2ec8 | 4K SDR | baba | 2 | c | nominal |
| f2455ea7e692 | 4K SDR | baba | 3 | e | nominal |
| c8abdcba49b9 | 4K PQ | abab | 0 | e | nominal |
| e97617023b9b | 4K PQ | abab | 1 | c | nominal |
| 570e06c3b17a | 4K PQ | abab | 2 | e | nominal |
| 0c4d6a8d318b | 4K PQ | abab | 3 | c | nominal |
| 4760813580bd | 4K PQ | baba | 0 | c | nominal |
| 61c5460ff908 | 4K PQ | baba | 1 | e | fair |
| 805251cf69a2 | 4K PQ | baba | 2 | c | fair |
| 590fb755ff79 | 4K PQ | baba | 3 | e | fair |

## Pixel 9 Pro

Vulkan, Mali-G715, driver `v1.r54p2-00eac0.7001668a7a05889fc53765c682b36a3f`, target `Rgba16Float`. The same matrix with `--energy` and `--cpu 7`. The report files do not record a git commit. ODPM is the energy source. The first report file is timestamped 3:34 PM EDT on 2 October 2026.

Thermal status is `none` and the recorded zone temperature is 48.0 °C, except two 4K PQ runs at `light` (same zone temperature): `4k-pq-e-3` (BABA rep 3) and `4k-pq-c-2` (BABA rep 2).

| Cell | Path | GPU p50 | GPU p99 | Host p50 | Host p99 |
|---|---|---:|---:|---:|---:|
| 1080p SDR | e | 0.883 | 1.041 | 1.477 | 2.164 |
| 1080p SDR | c | 6.216 | 6.618 | 9.169 | 10.857 |
| 1080p PQ | e | 1.450 | 1.811 | 2.096 | 2.590 |
| 1080p PQ | c | 5.390 | 6.260 | 3.224 | 10.748 |
| 4K SDR | e | 5.518 | 12.621 | 4.011 | 13.026 |
| 4K SDR | c | 13.832 | 14.880 | 19.638 | 22.745 |
| 4K PQ | e | 5.219 | 13.098 | 6.596 | 12.869 |
| 4K PQ | c | 13.823 | 14.292 | 20.226 | 21.675 |

e − c, same bootstrap:

| Cell | GPU p50 | GPU p99 | Host p50 | Host p99 |
|---|---|---|---|---|
| 1080p SDR | −5.313 [−5.346, −5.259] | −5.631 [−5.759, −5.506] | −7.462 [−7.767, −7.158] | −9.063 [−9.198, −8.908] |
| 1080p PQ | −4.044 [−4.542, −3.661] | −4.018 [−4.657, −3.557] | −2.281 [−5.728, −0.148] | −5.428 [−7.508, −3.958] |
| 4K SDR | −8.370 [−8.502, −8.296] | −3.798 [−7.636, −1.714] | −15.656 [−16.021, −15.354] | −11.981 [−15.049, −9.105] |
| 4K PQ | −8.459 [−8.943, −7.950] | −3.357 [−6.843, 0.128] | −13.527 [−13.788, −13.191] | −8.279 [−10.621, −5.937] |

Missed 120 Hz deadlines, by rep 0–3. Path `c` at 1080p PQ is split: rep 0 missed 237 frames and the other three missed 2, 0, and 0, which is why that host p50 interval is wide. The 4K PQ GPU p99 interval crosses zero.

| Cell | Path | Missed deadlines |
|---|---|---|
| 1080p SDR | e | 0, 0, 0, 0 |
| 1080p SDR | c | 227, 233, 227, 227 |
| 1080p PQ | e | 0, 0, 1, 0 |
| 1080p PQ | c | 237, 2, 0, 0 |
| 4K SDR | e | 15, 11, 1, 16 |
| 4K SDR | c | 238, 238, 238, 238 |
| 4K PQ | e | 40, 47, 41, 29 |
| 4K PQ | c | 238, 238, 238, 238 |

Steady memory, median bytes. Engine GPU bytes match the iPhone readings at the same size and path.

| Cell | Path | PSS | gpu_mem | engine GPU | wgpu reserved |
|---|---|---:|---:|---:|---:|
| 1080p SDR | e | 74569728 | 41893888 | 16598208 | 21102400 |
| 1080p SDR | c | 82073088 | 87130112 | 33186816 | 54787712 |
| 1080p PQ | e | 74058240 | 44986368 | 16598208 | 21102400 |
| 1080p PQ | c | 79680000 | 94965760 | 33186816 | 63176320 |
| 4K SDR | e | 87558144 | 109457408 | 66364608 | 71826496 |
| 4K SDR | c | 81101824 | 231178240 | 132719616 | 173504640 |
| 4K PQ | e | 88318976 | 122755072 | 66364608 | 71826496 |
| 4K PQ | c | 81192960 | 273199104 | 132719616 | 182458496 |

1080p PQ path `c` reserved 54787712 bytes on one run and 63176320 on the other three. The median is 63176320.

Energy is the median of the four runs' ODPM `joules_per_frame` and `watts`.

| Cell | Path | J/frame | Watts |
|---|---|---:|---:|
| 1080p SDR | e | 0.0122 | 1.467 |
| 1080p SDR | c | 0.0149 | 1.609 |
| 1080p PQ | e | 0.0164 | 1.968 |
| 1080p PQ | c | 0.0175 | 2.034 |
| 4K SDR | e | 0.0180 | 2.213 |
| 4K SDR | c | 0.0581 | 2.902 |
| 4K PQ | e | 0.0257 | 3.077 |
| 4K PQ | c | 0.0650 | 3.224 |

4K SDR path `e` rep 2 drew 0.0267 J/frame at 3.20 W. The other three runs of that cell and path were 0.0175–0.0182 J/frame.
