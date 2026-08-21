from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common.config import load_config
from layer4_speech_separation import (
    DirectionCountSpeakerClassifier,
    MossFormer2Backend,
    TigerBackend,
)
from layer4_speech_separation.offline import OfflineLayer4Pipeline, load_sealed_l3_tracks, persist_offline_results
from layer5_voice_classifier import Layer5Engine, NvidiaMarbleNetPlugin
from layer5_voice_classifier.gain_compensation import InputGainCompensationSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline L4 on one finalized runtime session")
    parser.add_argument("session_root", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--backend", choices=("mossformer2_ss_16k", "tiger_speech_16k"))
    parser.add_argument("--device", choices=("cuda", "cpu"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if not config.layer4.enabled:
        raise RuntimeError("Layer4 is disabled in project config")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    backend_id = args.backend or config.layer4.default_backend
    artifact = (
        config.layer4.mossformer2_artifact
        if backend_id == "mossformer2_ss_16k"
        else config.layer4.tiger_artifact
    )
    backend = (
        MossFormer2Backend(artifact, device=device)
        if backend_id == "mossformer2_ss_16k"
        else TigerBackend(artifact, device=device)
    )
    model_config = next(
        item for item in config.layer5.models if item.model_id == config.layer5.primary_model_id
    )
    l5 = Layer5Engine(
        NvidiaMarbleNetPlugin(
            model_config.model_id, model_config.model_artifact,
            device=device, window_spec=config.downstream_audio_window,
        ),
        threshold=config.layer5.voice_probability_limit,
        input_gain_compensation=InputGainCompensationSettings(
            **config.layer5.input_gain_compensation.model_dump()
        ),
        window_spec=config.downstream_audio_window,
    )
    pipeline = OfflineLayer4Pipeline(
        speaker_counter=DirectionCountSpeakerClassifier(),
        backends={backend_id: backend},
        layer5=l5,
        default_backend=backend_id,
    )
    sources = load_sealed_l3_tracks(args.session_root)
    if not sources:
        raise RuntimeError("finalized session contains no continuous L3 track audio")
    results = tuple(pipeline.process(source) for source in sources)
    manifest = persist_offline_results(args.session_root, results)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
