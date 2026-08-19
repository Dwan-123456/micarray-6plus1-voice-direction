# Third-party algorithm references

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
