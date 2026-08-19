# Third-party algorithm references

## Pyroomacoustics MUSIC / NormMUSIC

- Repository: https://github.com/LCAV/pyroomacoustics
- Reviewed files: `pyroomacoustics/doa/music.py`, `pyroomacoustics/doa/normmusic.py`, and `examples/doa_algorithms.py` on the upstream `master` branch during the 1.1.0 L2 design review.
- License: MIT (https://github.com/LCAV/pyroomacoustics/blob/master/LICENSE)
- Use: algorithmic and test reference for broadband MUSIC and per-frequency normalized MUSIC fusion.
- Runtime dependency: none; this project contains an independent rolling implementation specialized for 7 physical microphones, 2--4 kHz, and a 360-point grid.

## MUSIC and MDL papers

- R. O. Schmidt, “Multiple Emitter Location and Signal Parameter Estimation”: https://codar.com/images/about/1986Schmidt_MUSIC.pdf
- M. Wax and T. Kailath, “Detection of Signals by Information Theoretic Criteria”: https://doi.org/10.1109/TASSP.1985.1164557
- Use: mathematical references for the noise-subspace pseudo-spectrum and 0--3 source model-order selection. No paper source code is copied.

## Israel Cohen publications

- Publications index: https://israelcohen.com/publications/all-publications/
- Use: noise-estimation and robustness background only. No Cohen MUSIC open-source implementation was found, claimed, or copied.

## Voice-Separation-and-Enhancement

- Repository: https://github.com/KyleZhang1118/Voice-Separation-and-Enhancement
- Reviewed commit: `77d16c120356dbbca3ee768d293df5d743d343ad`
- Upstream language: MATLAB
- Use: algorithmic reference for the 6+1 circular-array STFT flow and WNG-constrained loading search.
- Runtime dependency: none; no upstream MATLAB objects cross project interfaces.
- License: no license file was visible in the reviewed repository. Upstream source is therefore not copied or redistributed; this project contains an independent implementation of the described equations and architecture.

## ODAS

- Repository: https://github.com/introlab/odas
- Reviewed commit: `bcb845434495e293df3d48f1203b7a86e1852449`
- License: MIT
- Use: design reference for microphone-array sound-source tracking, Kalman state estimation, track confidence, and inactive-track lifetime.
- Runtime dependency: none; the Test UI tracker is an independent Python implementation.

## Roboflow Trackers

- Repository: https://github.com/roboflow/trackers
- Reviewed commit: `54888a4aa7b9678481dc2d1f1d1d0d38906706d2`
- License: Apache-2.0
- Use: design reference for the tracking-by-detection sequence of prediction, gated association, update, creation, and track retirement.
- Runtime dependency: none; visual bounding-box tracking code is not imported or copied.
