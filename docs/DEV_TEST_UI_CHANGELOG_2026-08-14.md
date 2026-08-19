# Development Test UI change record — 2026-08-14

> 历史快照：本文只记录2026-08-14当时某次变更的范围，不代表当前项目状态，也不得作为实施或完成度依据。文中L2 Noise Estimation、Mean/Std Gate或SNR Gate均已被v0.3的L1概率输出与L2 Probability Gate替换。当前状态以根规格、`config/config.yaml`和各层README为准。

Scope: completed code changes for the Development Test UI before Layer 4. Layer 4 remains unavailable and was not modified.

## Current code changes

### L2 to UI window-consistent data flow

- `DevUiFrame` now carries the window-aligned Noise Estimation result, Gate decision, applied candidate threshold, iterative-search setting, and scan-configuration revision as one snapshot.
- The aggregator clears retained L2/L3 data when session or epoch changes, including Gate-blocked frames that have no SRP response.
- A new L2 result supersedes old previews and tracked audio so stale candidate/L3 rows cannot be mixed with the current window.
- The UI displays NE `noise_mean_db`, `noise_std_db`, NE state, and Gate state from the same decision window.
- Candidate threshold and iterative-search status are rendered from the settings actually applied to the displayed window rather than only from the latest control value.
- The documented default Gate is restored to `snr_hysteresis_v1` (8/4 dB, 2-frame attack, 10-frame release); `mean_std_v1` remains an explicit fallback implementation. NE mean/std remain diagnostic UI outputs and do not replace the formal Gate.

### Formal L3 visualization and candidate linkage

- The UI now consumes formal `BeamformPreview` objects instead of discarding them.
- The L3 panel renders the selected candidate's real 320 ms waveform and real `[33,169]` spectrogram.
- Selecting a candidate in the L2 polar plot or candidate table selects and freezes the matching L3 preview by `window_id` and angle.
- The preview header displays the L3 runtime backend, fallback reason, preprocessing version, window, and angle.
- Formal 320 ms preview playback and explicit stop controls are connected.

### Playback safety and temporary storage

- Disk-backed tracked-audio playback now applies DC removal, configured peak normalization, volume scaling, and boundary fades without loading the complete file into memory.
- Playback snapshots can be deleted automatically when released.
- Tracked L3 audio uses a bounded rolling disk cache; reset removes orphaned live/playback files.
- Ended tracks have count and retention limits, and expired cache files are deleted.
- Long Gate gaps cannot directly revive an expired audio ID.

### Recording and shutdown resilience

- Scratch-recorder command queue failures now enter an explicit error state instead of remaining indefinitely in `finalizing` or `paused`.
- Runtime recording retains the writer exception, detects failed writer shutdown, uses a bounded join, and records `incomplete`/`corrupt` manifest state instead of waiting forever after a writer failure.
- CUDA timing now synchronizes completed L3 work before publishing compute/latency values.

### Regression coverage added to the source tree

- Epoch transition and stale L2/L3 clearing.
- Gate-blocked NE publication and candidate-table clearing.
- Formal L3 preview visibility and UI linkage.
- Bounded disk cache, reset cleanup, ended-track limits/TTL, and ID expiry across Gate gaps.
- Scratch command queue failure cleanup.
- Runtime-recording writer failure with a full queue.

## Not completed in this snapshot

- Physical light commands were not exercised; light-command timeout/error UI handling still needs a later pass.
- Active processing-queue drops at capture stop still need explicit result-watermark accounting.
- Playback state/error signals and the final close-path `try/finally` hardening still need a later pass.
- No Layer 4 model, inference, metric, or UI behavior was added.

## Verification status

At the user's request, no further software test run, UI launch, microphone acquisition, serial-light command, or hardware validation was performed after this snapshot was recorded. The current final file state is therefore recorded but not certified by a post-change test run.
