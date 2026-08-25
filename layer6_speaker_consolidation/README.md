# L6 speaker consolidation

L6 is a manual, offline stage. It accepts only completed L4 results after every
retained L4 branch has received aligned L5 probabilities. Long L5 speech runs
are assigned in 500 ms pieces while CAMPPlus retains up to 1.5 s of context
inside the same voice region, so one L2 track can change speaker without
requiring a long silence. Constrained
average-link clustering uses sustained simultaneous L2 directions as negative
evidence while allowing near-identical cross-track leakage copies. Short speech
residuals attach to reliable clusters instead of creating phantom identities.
Overlapping copies retain the higher-quality source. Outputs remain 16 kHz mono
and carry their absolute 48 kHz capture-timeline bounds.

The quality score is fixed at 30% L5 voice confidence, 30% speaker-centroid
similarity, 20% DNSMOS, 10% segmental SNR, and 10% continuity. L6 never runs
automatically during capture and never changes the upstream L2 track ID or angle.
