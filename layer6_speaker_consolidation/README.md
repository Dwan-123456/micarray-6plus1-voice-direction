# L6 speaker consolidation

L6 is a manual offline stage that runs only after unmerged L4 A/B tracks have
received aligned L5 decisions. It performs complete-track voiceprint clustering;
it no longer divides tracks into 1.5-second speaker-analysis fragments.

For every L4 source, candidate A is always selected for complete-track CAMPPlus
embedding. Candidate B is embedded only when all three gates
pass: the absolute A/B L4 match-score gap is at most `0.20`, B's match score is
strictly greater than `0.50`, and B's normalized L4 DNSMOS score is strictly
greater than `0.30`. Tracks with no L5 voice are still embedded as required, but
a cluster containing no L5-active audio cannot create an empty display output.

All selected embeddings are compared pairwise. Average-link AHC uses the
configured cosine threshold (`0.62`) and forcibly limits the result to at most
three session-local voiceprints. Each voiceprint owns one or more complete L4
audio tracks; `Layer6Result.metadata.voiceprint_audio_ids` records this one-to-many
relationship, while the full symmetric similarity matrix remains available for
audit.

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
