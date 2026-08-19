from __future__ import annotations

import tomllib
from pathlib import Path

from setuptools import find_namespace_packages


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_distribution_includes_runtime_and_its_first_party_dependencies():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)

    include = project["tool"]["setuptools"]["packages"]["find"]["include"]
    packages = set(find_namespace_packages(PROJECT_ROOT, include=include))

    assert "app" in packages
    assert "config" in packages
    assert "data_management" in packages
    assert {
        "common",
        "gui.dev_test_ui",
        "ingest",
        "layer1_input",
        "layer2_source_detection",
        "layer3_direction_signal",
        "layer4_voice_classifier",
        "windowing",
    } <= packages
    assert project["tool"]["setuptools"]["package-data"]["config"] == ["*.yaml"]
    assert (PROJECT_ROOT / "config" / "config.yaml").is_file()
