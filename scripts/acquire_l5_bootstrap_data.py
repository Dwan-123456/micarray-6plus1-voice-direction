from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import urllib.request
import wave
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "data" / "external_sources"
DOWNLOAD_DATE = datetime.now(timezone.utc).date().isoformat()
USER_AGENT = "micarray-l5-data-acquisition/1.0"


REGISTRY = [
    ("AliMeeting", "OpenSLR 119", "https://www.openslr.org/119/", "CC BY-SA 4.0", "first", "deferred", "smallest combined audio package is 3.42 GB"),
    ("AISHELL-4", "release page", "https://www.aishelltech.com/aishell_4", "package license requires verification", "second", "deferred", "license/package access must be verified before download"),
    ("AMI Meeting Corpus", "manual annotations 1.6.2", "https://groups.inf.ed.ac.uk/ami/download/", "CC BY 4.0", "first", "selected", "one meeting mix plus manual time annotations"),
    ("AVA-Speech", "v1.0 labels", "https://sites.research.google/gr/ava/download/", "CC BY 4.0 labels; source video rights separate", "first", "labels_only", "labels downloaded; source media deliberately excluded"),
    ("VoxConverse", "release page", "https://mmai.io/datasets/voxconverse/", "annotations open; original video rights retained", "third", "deferred", "media copyright boundary"),
    ("LibriCSS", "release page", "https://github.com/chenzhuo1011/libri_css", "audio terms require verification", "second", "deferred", "code license cannot substitute for audio terms"),
    ("DiPCo", "CHiME release", "https://arxiv.org/abs/1909.13447", "release package license requires verification", "third", "deferred", "package-specific terms not yet captured"),
    ("STARSS23", "Zenodo 7880637", "https://zenodo.org/records/7880637", "record/package terms require file-level capture", "second", "deferred", "7.5 hours is outside bootstrap size"),
    ("AISHELL-1", "OpenSLR 33", "https://www.openslr.org/33/", "Apache 2.0", "first", "metadata_only", "15 GB audio deferred; supplementary resource may be fetched later"),
    ("Mozilla Common Voice", "version not selected", "https://commonvoice.mozilla.org/en/datasets", "verify per release", "second", "deferred", "version and Chinese subset must be selected"),
    ("LibriSpeech", "OpenSLR 12", "https://www.openslr.org/12/", "CC BY 4.0", "second", "deferred", "full archives unnecessary for bootstrap"),
    ("DNS Challenge", "DNS5 repository", "https://github.com/microsoft/DNS-Challenge", "mixed upstream licenses", "first", "deferred", "license-clear upstream subset not yet isolated"),
    ("MUSAN", "OpenSLR 17", "https://www.openslr.org/17/", "CC BY 4.0", "first", "deferred", "11 GB archive; music/noise require speech screening"),
    ("FSD50K", "Zenodo 4060432", "https://zenodo.org/records/4060432", "per-file CC0/CC BY/CC BY-NC/CC Sampling+", "second", "metadata_only", "audio requires per-file commercial whitelist and speech screening"),
    ("DEMAND", "1.0 / Zenodo 1227121", "https://zenodo.org/records/1227121", "manage as CC BY-SA 3.0 pending metadata conflict resolution", "second", "selected", "one 48 kHz environment; candidate negatives require speech screening"),
    ("OpenSLR RIR and Noise", "OpenSLR 28", "https://www.openslr.org/28/", "Apache 2.0", "first", "deferred", "1.3 GB archive; source recorded for later synthesis batch"),
    ("AudioSet", "label ontology only", "https://research.google.com/audioset/download.html", "labels only; YouTube media not openly licensed", "metadata", "labels_only_later", "never infer non-speech from missing Speech label"),
    ("WHAM! / ESC-50", "not selected", "", "contains non-commercial restrictions", "excluded", "excluded", "not for production training if commercial use is possible"),
    ("AVSpeech / VoxCeleb / WenetSpeech", "not selected", "", "media/application boundaries", "third", "deferred", "large or access-constrained"),
    ("TVSM", "features only", "", "not assessed", "excluded", "excluded", "no compatible raw audio"),
]


DOWNLOADS = [
    ("ami", "raw/ami_public_manual_1.6.2.zip", "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip", 22_887_865),
    ("ami", "raw/ES2008a.Mix-Headset.wav", "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2008a/audio/ES2008a.Mix-Headset.wav", 33_387_564),
    ("ami", "licenses/source_page.html", "https://groups.inf.ed.ac.uk/ami/download/", None),
    ("ami", "licenses/CC-BY-4.0.html", "https://creativecommons.org/licenses/by/4.0/legalcode.en", None),
    ("ava_speech", "raw/ava_speech_labels_v1.csv", "https://research.google.com/ava/download/ava_speech_labels_v1.csv", 1_635_390),
    ("ava_speech", "licenses/source_page.html", "https://sites.research.google/gr/ava/download/", None),
    ("ava_speech", "licenses/CC-BY-4.0.html", "https://creativecommons.org/licenses/by/4.0/legalcode.en", None),
    ("demand", "raw/NFIELD_48k.zip", "https://zenodo.org/api/records/1227121/files/NFIELD_48k.zip/content", 270_538_724),
    ("demand", "raw/DEMAND.pdf", "https://zenodo.org/api/records/1227121/files/DEMAND.pdf/content", 87_837),
    ("demand", "licenses/zenodo_record.json", "https://zenodo.org/api/records/1227121", None),
    ("demand", "licenses/CC-BY-SA-3.0.html", "https://creativecommons.org/licenses/by-sa/3.0/legalcode.en", None),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, expected_size: int | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise OSError(f"下载大小不符：需要 {expected_size} 字节，实际 {partial.stat().st_size} 字节")
    partial.replace(target)


def safe_extract(archive: Path, target: Path) -> None:
    marker = target / ".extraction_complete"
    if marker.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (target / member.filename).resolve()
            if root not in destination.parents and destination != root:
                raise ValueError(f"unsafe archive member: {member.filename}")
        bundle.extractall(target)
    marker.write_text(f"archive_sha256={sha256(archive)}\n", encoding="utf-8")


def audio_info(path: Path) -> dict[str, int | float | str] | None:
    try:
        with wave.open(str(path), "rb") as source:
            frames = source.getnframes()
            rate = source.getframerate()
            return {
                "sample_rate": rate,
                "channels": source.getnchannels(),
                "sample_width_bytes": source.getsampwidth(),
                "duration_seconds": frames / rate if rate else 0,
            }
    except (wave.Error, EOFError):
        return None


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-only", action="store_true", help="不联网，只核验现有文件并生成清单")
    args = parser.parse_args(argv)
    ROOT.mkdir(parents=True, exist_ok=True)
    registry = [
        dict(zip(("dataset_name", "version", "source_url", "license_name", "batch", "status", "decision_reason"), row))
        for row in REGISTRY
    ]
    write_json(ROOT / "registry" / "source_registry.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "sources": registry})

    failures = []
    for dataset, relative, url, expected_size in DOWNLOADS:
        target = ROOT / dataset / relative
        valid_size = target.exists() and target.stat().st_size and (
            expected_size is None or target.stat().st_size == expected_size
        )
        if valid_size:
            continue
        if args.finalize_only:
            failures.append(
                {
                    "dataset": dataset,
                    "url": url,
                    "target": str(target.relative_to(ROOT)),
                    "error": f"missing_or_incomplete; expected_size={expected_size}",
                }
            )
            continue
        try:
            download(url, target, expected_size)
        except Exception as exc:  # preserve every failure in the required report
            failures.append({"dataset": dataset, "url": url, "target": str(target.relative_to(ROOT)), "error": repr(exc)})

    ami_archive = ROOT / "ami" / "raw" / "ami_public_manual_1.6.2.zip"
    demand_archive = ROOT / "demand" / "raw" / "NFIELD_48k.zip"
    if ami_archive.exists():
        safe_extract(ami_archive, ROOT / "ami" / "extracted" / "manual_annotations_1.6.2")
    if demand_archive.exists() and demand_archive.stat().st_size == 270_538_724 and zipfile.is_zipfile(demand_archive):
        safe_extract(demand_archive, ROOT / "demand" / "extracted" / "NFIELD_48k")
    else:
        failures.append(
            {
                "dataset": "demand",
                "url": "https://zenodo.org/api/records/1227121/files/NFIELD_48k.zip/content",
                "target": "demand/raw/NFIELD_48k.zip",
                "error": "archive_not_complete_or_not_a_valid_zip",
            }
        )

    manifests = {
        "ami": {
            "dataset_name": "AMI Meeting Corpus",
            "version": "manual annotations 1.6.2 / ES2008a",
            "source_url": "https://groups.inf.ed.ac.uk/ami/download/",
            "download_date": DOWNLOAD_DATE,
            "license_name": "CC BY 4.0",
            "commercial_use": True,
            "modification": True,
            "redistribution": True,
            "label_strength": "strong word/segment time annotations",
            "group_fields": ["meeting", "session", "speaker"],
            "possible_unlabelled_speech": False,
            "recommended_use": ["train", "external stress"],
            "notes": "Do not label the complete meeting as voice; derive intervals from manual annotations and preserve pauses.",
        },
        "ava_speech": {
            "dataset_name": "AVA-Speech",
            "version": "1.0 labels",
            "source_url": "https://sites.research.google/gr/ava/download/",
            "download_date": DOWNLOAD_DATE,
            "license_name": "CC BY 4.0 for labels; source video copyright remains separate",
            "commercial_use": "labels yes with attribution; media requires separate rights review",
            "modification": True,
            "redistribution": "labels yes with attribution; do not redistribute source media by assumption",
            "label_strength": "strong interval labels",
            "group_fields": ["video_id", "episode"],
            "possible_unlabelled_speech": False,
            "recommended_use": ["external stress retrieval"],
            "notes": "Labels only. No YouTube/movie audio downloaded in this batch.",
        },
        "demand": {
            "dataset_name": "DEMAND",
            "version": "1.0 / Zenodo 1227121 / NFIELD 48 kHz",
            "source_url": "https://zenodo.org/records/1227121",
            "download_date": DOWNLOAD_DATE,
            "license_name": "CC BY-SA 3.0 strict handling; Zenodo structured metadata also reports CC BY 4.0",
            "commercial_use": True,
            "modification": True,
            "redistribution": True,
            "label_strength": "environment-level weak label",
            "group_fields": ["environment", "recording", "simultaneous_channel_set"],
            "possible_unlabelled_speech": True,
            "recommended_use": ["synthesis candidate", "external stress"],
            "notes": "Do not mark as non-voice until all 16 synchronized channels have been screened for speech.",
        },
    }

    rows = []
    for path in sorted(
        p
        for p in ROOT.rglob("*")
        if p.is_file() and not p.name.endswith(".partial") and "failed" not in p.relative_to(ROOT).parts
    ):
        if path.name in {"files.csv", "batch_manifest.json"}:
            continue
        info = audio_info(path) if path.suffix.lower() == ".wav" else None
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256(path),
                "sample_rate": "" if info is None else info["sample_rate"],
                "channels": "" if info is None else info["channels"],
                "duration_seconds": "" if info is None else round(float(info["duration_seconds"]), 6),
            }
        )
    for dataset, manifest in manifests.items():
        selected = [row for row in rows if row["path"].startswith(dataset + "/")]
        manifest["archives"] = [
            {"archive_name": Path(row["path"]).name, "size": row["size"], "sha256": row["sha256"]}
            for row in selected
            if "/raw/" in row["path"]
        ]
        audio_rows = [row for row in selected if row["sample_rate"] != ""]
        manifest["audio_inventory"] = audio_rows
        manifest["total_audio_duration_seconds"] = round(sum(float(row["duration_seconds"]) for row in audio_rows), 3)
        license_rows = [row for row in selected if "/licenses/" in row["path"]]
        manifest["license_snapshots"] = [
            {"path": row["path"], "sha256": row["sha256"]} for row in license_rows
        ]
        write_json(ROOT / dataset / "source_manifest.json", manifest)

    # Re-scan so source manifests themselves are covered by the root batch manifest.
    rows = []
    for path in sorted(
        p
        for p in ROOT.rglob("*")
        if p.is_file() and not p.name.endswith(".partial") and "failed" not in p.relative_to(ROOT).parts
    ):
        if path.name in {"files.csv", "batch_manifest.json"}:
            continue
        info = audio_info(path) if path.suffix.lower() == ".wav" else None
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256(path),
                "sample_rate": "" if info is None else info["sample_rate"],
                "channels": "" if info is None else info["channels"],
                "duration_seconds": "" if info is None else round(float(info["duration_seconds"]), 6),
            }
        )
    with (ROOT / "files.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        ROOT / "batch_manifest.json",
        {
            "schema_version": "external_audio_batch_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "L5 bootstrap acquisition only; no model training and no dataset split",
            "file_count": len(rows),
            "total_bytes": sum(int(row["size"]) for row in rows),
            "files_csv_sha256": sha256(ROOT / "files.csv"),
            "sources": list(manifests),
        },
    )
    write_json(ROOT / "failed_downloads.json", failures)
    print(json.dumps({"files": len(rows), "bytes": sum(int(row["size"]) for row in rows), "failures": failures}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
