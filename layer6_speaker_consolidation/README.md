# L6 speaker consolidation

L6 is a manual, offline stage. It accepts only completed L4 results after every
retained L4 branch has received aligned L5 probabilities. CAMPPlus embeds valid
speech regions, average-link hierarchical clustering assigns session-local
Speaker A/B/C identities, and overlapping copies of the same speaker retain the
higher-quality source. Outputs remain 16 kHz mono and carry their absolute 48 kHz
capture-timeline bounds.

The quality score is fixed at 30% L5 voice confidence, 30% speaker-centroid
similarity, 20% DNSMOS, 10% segmental SNR, and 10% continuity. L6 never runs
automatically during capture and never changes the upstream L2 track ID or angle.
