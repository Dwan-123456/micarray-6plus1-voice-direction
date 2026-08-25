# L6 speaker consolidation

L6 has a throttled provisional preview in the `1.3.5` development runtime and an
authoritative offline stage after a sealed Test UI L4/L5 batch. The preview runs
on first evidence, topology changes, or a 30-second watermark interval; unchanged
two-second CAMPPlus evidence is cached. The sealed batch replaces the preview. It consumes unmerged L4
A/B tracks plus any single-speaker bypass track that is the source's only A track.
It concatenates each selected track's L5 Voice
frames, divides that continuous speech into fixed two-second segments, and extracts one
CAMPPlus embedding per segment. Upstream chunks may be any integer 3..15 seconds:
two-second segmentation crosses those boundaries, so a residual is retained and
joined with future evidence; the final residual is used when it has at least 500 ms.
Same-length segments are inferred in batches.

For every L4 source, candidate A is always selected for segmented CAMPPlus
embedding. Candidate B is embedded only when all three gates
pass: the absolute A/B L4 match-score gap is at most `0.20`, B's match score is
strictly greater than `0.50`, and B's normalized L4 DNSMOS score is strictly
greater than `0.30`. Tracks with less than 500 ms of cumulative L5 Voice do not
create a voiceprint.

Within one track, every segment's median similarity to the other segments is its
centrality. Robust low-centrality outliers are discarded, while at least the two
most central segments remain. Two tracks' segment embeddings are paired globally
one-to-one with the Hungarian algorithm. The track score is the weakest of
the top evidence set, where the evidence set contains at least one pair and at
least 30 percent of the shorter track's retained segments. A one-segment short
track can therefore merge when that segment reaches the same threshold. Complete-link AHC
requires every cross-track pair in two clusters to pass the configured cosine
threshold (`0.62`), preventing a B track from bridging two incompatible people.
The result is forcibly limited to at most three session-local voiceprints. Each
voiceprint owns one or more complete L4 audio tracks;
`Layer6Result.metadata.voiceprint_audio_ids` records this one-to-many relation,
while segment counts, evidence counts, the symmetric track-score matrix and each
pair's median, 25th percentile, threshold coverage, mutual-nearest-neighbour
coverage, representative cosine, MAD and standard deviation remain available
for audit. `LogisticCalibration` can convert these features to an interpretable
same-speaker probability only after fitting with real speaker-identity labels;
L6 does not manufacture labels or ship an unverified probability calibration.

For each voiceprint, associated tracks are projected onto the recording's
absolute 48 kHz capture bounds. Only L5-active 20 ms frames are inserted. Tracks
are ordered by L4 MOS, then deterministically by match score and voiceprint
similarity; an occupied frame is never overwritten, so overlaps retain the
higher-MOS audio. Missing frames remain silent.

After the absolute-time merge, all leading and trailing silence is removed.
Every internal silent run longer than two seconds is shortened to two seconds;
shorter pauses remain unchanged. The displayed 16 kHz Speaker A-E waveform is
therefore silence-compressed and is not required to remain equal to the original
recording duration. Its source recording bounds remain attached for audit.

The Test UI marks live speaker rows as provisional revisions; their numbering may
change as evidence arrives. Only the sealed canonical result is final. Both views
show linked tracks, source L2 IDs, mean L4 MOS, compressed duration, waveform and playback.
