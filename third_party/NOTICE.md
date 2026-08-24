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

## CountNet speaker-count CRNN

- Repository: https://github.com/faroit/CountNet
- Vendored revision: `ae15e1ac096862667a7bfdedf0b67a70a7543edd`
- Upstream assets: `models/CRNN.h5` and `models/scaler.npz`, with hashes pinned in the local model manifest.
- License: MIT.
- Use: optional Test-UI-only L1 estimate of zero, one, or two-or-more concurrent speakers from a rolling
  five-second Center Mic context. The old Keras 1.2.2/Theano inference graph is deterministically ported to
  TorchScript; the upstream HDF5 files are not redistributed.

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

## ClearerVoice-Studio / MossFormer2

- Repository: https://github.com/modelscope/ClearerVoice-Studio
- Vendored revision: `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`
- Model: https://huggingface.co/alibabasglab/MossFormer2_SS_16K, revision `407cb030cd66340918ebb6c8cc63b18f8592cdbe`
- License: Apache-2.0
- Use: optional offline two-speaker comparison backend. The inference-only source snapshot and hash-pinned weights are redistributed with their license and manifest.

## TIGER speech separation

- Repository: https://github.com/JusperLee/TIGER
- Vendored revision: `9f18d4a10a7137e1ce8052cfb62215179f1287b6`
- Model: https://huggingface.co/JusperLee/TIGER-speech
- License: Apache-2.0
- Use: optional offline two-speaker comparison backend. The inference-only source snapshot and hash-pinned weights are redistributed with their license and manifest.
