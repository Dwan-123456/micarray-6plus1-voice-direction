# L6 Hungarian matching experiment — 2026-08-25

> The evaluated implementation, including the one-segment follow-up, was later
> adopted on `codex/develop-v1.3.3`. This document retains the experiment's
> measurements and limitations as its audit record.

## Follow-up: allow one-segment short tracks

The hard minimum of two matched segments was removed after the initial
comparison. Evidence coverage remains 30 percent of the shorter track, with an
absolute minimum of one pair. A short track with one retained voiceprint can now
merge at the normal `0.62` threshold.

The newest cache WAV directory had already been cleaned by the Test UI before
this follow-up run. Applying the revised rule to its previously captured matrix
changes single-segment decisions from `-1` to their measured cosine; the short A
track has `0.819/0.784` links to an established cluster. The remaining older
cache was reprocessed and stayed at three clusters because its short track's
best cosine was only `0.410`. A fresh recording is still required to confirm the
new final speaker count through the real B-candidate admission gate.

## Scope

This experiment preserves `codex/develop-v1.3.3` at commit `5b15fa3` and changes
only the track-to-track segment matching layer on
`codex/l6-hungarian-matching-experiment`. L5 Voice selection, two-second
CAMPPlus extraction, within-track outlier removal, complete-link clustering,
the `0.62` threshold and the maximum of three speakers are unchanged.

The old matcher greedily selected the largest remaining cosine. The experiment
uses a globally optimal one-to-one Hungarian assignment and records the selected
score, median, 25th percentile, threshold coverage, mutual-nearest-neighbour
coverage, representative-vector cosine, MAD, standard deviation and evidence
counts. A logistic calibration type and labeled-data fitter are included, but no
probability model is fitted because the available corpus labels contain source
count/direction rather than speaker identity.

## Read-only cache comparison

No cache WAV or runtime data is committed. Both available non-empty L4 caches
were reprocessed with the repository MarbleNet and CAMPPlus artifacts. The WAV
cache does not persist A/B match-score metadata, so the comparison covers every
cached WAV rather than reconstructing the B-candidate admission gate. A tracks
are always admitted by L6, so the short-A-track finding below is unaffected.

| Cache | usable tracks | track pairs | changed pair scores | greedy clusters | Hungarian clusters |
|---|---:|---:|---:|---:|---:|
| `micarray_dev_ui_l4_b5t2bu42` | 7 | 21 | 2 | 3 | 3 |
| `micarray_dev_ui_l4_s15ureb1` | 4 | 6 | 1 | 3 | 3 |

Changed scores:

- newest cache, `3B ↔ 6A`: `0.680659 → 0.665326`;
- newest cache, `3B ↔ 12A`: `0.527730 → 0.628444`;
- older cache, track `11 ↔ 12`: `0.641963 → 0.631638`.

Across 27 pairs, Hungarian assignment repaired one threshold-crossing greedy
pair but did not change either cache's final cluster count. On the newest cache,
several approximately three-second tracks yielded only one retained two-second
voiceprint. Both rules require at least two matched segments, so those tracks
receive decision score `-1` even when their single cosine is high (for example
`0.819`). This is the direct reason changing the assignment algorithm alone
cannot remove the extra ID.

## Performance

The comparison repeated matching 200 times after embeddings were available.
Greedy matching cost `0.009–0.014 ms` per pair; Hungarian matching plus all
features cost `0.073–0.089 ms` per pair. The relative increase is roughly
6–8 times, but the absolute increase is below `0.1 ms` per track pair and is
negligible beside VAD and CAMPPlus inference.

## Conclusion

Hungarian assignment is a sound, auditable replacement for greedy pairing, but
the observed L6 over-segmentation is not primarily a pairing-optimization bug.
The follow-up now permits one-segment tracks at the standard threshold. This
removes the deterministic rejection but increases sensitivity to one accidental
high cosine; a fresh two-speaker recording is needed to measure that trade-off.
A real logistic `P(same speaker)` must wait for labeled same/different speaker
pairs; fitting it from source count alone would create false ground truth.
