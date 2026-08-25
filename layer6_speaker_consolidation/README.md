# L6 speaker consolidation

L6 is a manual offline stage that runs only after unmerged L4 A/B tracks have
received aligned L5 decisions. It concatenates each selected track's L5 Voice
frames, divides that speech into fixed two-second segments, and extracts one
CAMPPlus embedding per segment. A final residual is retained only when it has at
least 500 ms of speech. Same-length segments are inferred in batches.

For every L4 source, candidate A is always selected for segmented CAMPPlus
embedding. Candidate B is embedded only when all three gates
pass: the absolute A/B L4 match-score gap is at most `0.20`, B's match score is
strictly greater than `0.50`, and B's normalized L4 DNSMOS score is strictly
greater than `0.30`. Tracks with less than 500 ms of cumulative L5 Voice do not
create a voiceprint.

Within one track, every segment's median similarity to the other segments is its
centrality. Robust low-centrality outliers are discarded, while at least the two
most central segments remain. On the experimental branch, two tracks' segment
embeddings are paired globally one-to-one with the Hungarian algorithm. The track score is the weakest of
the top evidence set, where the evidence set contains at least two pairs and at
least 30 percent of the shorter track's retained segments. Complete-link AHC
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
shorter pauses remain unchanged. The displayed 16 kHz Speaker A/B/C waveform is
therefore silence-compressed and is not required to remain equal to the original
recording duration. Its source recording bounds remain attached for audit.

The Test UI displays one row per clustered voiceprint with its number of linked
tracks, source L2 IDs, mean L4 MOS, compressed duration, waveform and playback.
