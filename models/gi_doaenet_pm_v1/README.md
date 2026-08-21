# GI-DOAEnet PM runtime asset

L2 can use the upstream GI-DOAEnet PM model as a complete alternative chain:
GI-DOAEnet probabilities, circular candidate filtering, LMB/JPDA association,
circular Kalman state and the existing public L2 DTO.

The upstream repository does not publish a license file at the pinned revision,
so its source and checkpoint are intentionally not committed here. Install the
pinned files locally with:

```powershell
.\.venv\Scripts\python.exe scripts\install_gi_doaenet.py --acknowledge-upstream-terms
```

The installer verifies the archive revision and PM checkpoint SHA-256. Runtime
loading is lazy: MUSIC startup is unaffected, while selecting the NN chain in
Development Test UI loads the model on its first accepted Gate window.
