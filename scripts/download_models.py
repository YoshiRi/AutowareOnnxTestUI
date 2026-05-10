#!/usr/bin/env python3
"""
Download Autoware ML models required by AutowareOnnxTestUI.

Source:
  https://github.com/autowarefoundation/autoware/blob/main/ansible/roles/artifacts/tasks/main.yaml

Usage:
  # Download all Phase-1 image models (default model root)
  uv run python scripts/download_models.py

  # Specify a custom destination directory
  uv run python scripts/download_models.py --model-root ~/autoware_models

  # List available model groups without downloading
  uv run python scripts/download_models.py --list

  # Download specific groups only
  uv run python scripts/download_models.py --models tensorrt_yolox traffic_light_classifier

The destination for each file is: <model-root>/<group>/<filename>
This matches the model_dir layout expected by model_registry.yaml.
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Model catalogue
# Source: ansible/roles/artifacts/tasks/main.yaml (Phase-1 image models)
# ---------------------------------------------------------------------------

MODELS: dict[str, list[dict]] = {
    "tensorrt_yolox": [
        {
            "file": "yolox-sPlus-T4-960x960-pseudo-finetune.onnx",
            "url": "https://awf.ml.dev.web.auto/perception/models/object_detection_yolox_s/v1/yolox-sPlus-T4-960x960-pseudo-finetune.onnx",
            "sha256": "f5054e8a890c3be86dc1b4b89a5a36fb2279d4f6110b0159e793be062641bf65",
            "desc": "YOLOX-S T4 960×960  detection only",
        },
        {
            "file": "yolox-sPlus-opt-pseudoV2-T4-960x960-T4-seg16cls.onnx",
            "url": "https://awf.ml.dev.web.auto/perception/models/object_detection_semseg_yolox_s/v1/yolox-sPlus-opt-pseudoV2-T4-960x960-T4-seg16cls.onnx",
            "sha256": "73b3812432cedf65cebf02ca4cb630542fc3b1671c4c0fbf7cee50fa38e416a8",
            "desc": "YOLOX-S multi-head  detection + semantic segmentation (16 classes)",
        },
        {
            "file": "yolox-tiny.onnx",
            "url": "https://awf.ml.dev.web.auto/perception/models/yolox-tiny.onnx",
            "sha256": "471a665f4243e654dff62578394e508db22ee29fe65d9e389dfc3b0f2dee1255",
            "desc": "YOLOX-tiny  lightweight detection",
        },
        {
            "file": "label.txt",
            "url": "https://awf.ml.dev.web.auto/perception/models/label.txt",
            "sha256": "3540a365bfd6d8afb1b5d8df4ec47f82cb984760d3270c9b41dbbb3422d09a0c",
            "desc": "Class labels (8 classes)",
        },
        {
            "file": "semseg_color_map.csv",
            "url": "https://awf.ml.dev.web.auto/perception/models/object_detection_semseg_yolox_s/v1/semseg_color_map.csv",
            "sha256": "3d93ca05f31b63424d7d7246a01a2365953705a0ed3323ba5b6fddd744a4bfea",
            "desc": "Semantic segmentation color map (16 classes)",
        },
    ],
    "traffic_light_classifier": [
        {
            "file": "traffic_light_classifier_mobilenetv2_batch_1.onnx",
            "url": "https://awf.ml.dev.web.auto/perception/models/traffic_light_classifier/v4/traffic_light_classifier_mobilenetv2_batch_1.onnx",
            "sha256": "455b71b3b20d3a96aa0e49f32714ba50421f668a2f9b9907c30b1346ac8a3703",
            "desc": "Traffic light classifier MobileNetV2 batch=1",
        },
        {
            "file": "traffic_light_lamp_recognizer_comlops.onnx",
            "url": "https://awf.ml.dev.web.auto/perception/models/traffic_light_classifier/v4/traffic_light_lamp_recognizer_comlops.onnx",
            "sha256": "300af04a5893961195a9b7ffd9d80364280f328a6dc549f9b1e05d428d96d2e5",
            "desc": "Traffic light lamp recognizer (CompLOPS)",
        },
        {
            "file": "lamp_labels.txt",
            "url": "https://awf.ml.dev.web.auto/perception/models/traffic_light_classifier/v4/lamp_labels.txt",
            "sha256": "1a5a49eeec5593963eab8d70f48b8a01bfb07e753e9688eb1510ad26e803579d",
            "desc": "Car traffic light labels",
        },
        {
            "file": "lamp_labels_ped.txt",
            "url": "https://awf.ml.dev.web.auto/perception/models/traffic_light_classifier/v4/lamp_labels_ped.txt",
            "sha256": "5427e1b7c2e33acd9565ede29e77992c38137bcf7d7074c73ebbc38080c6bcac",
            "desc": "Pedestrian traffic light labels",
        },
        {
            "file": "lamp_recognizer_ml.param.yaml",
            "url": "https://awf.ml.dev.web.auto/perception/models/traffic_light_classifier/v4/lamp_recognizer_ml.param.yaml",
            "sha256": "80d92943a036ac454aac200a3be83d693dacab924c869f556ef09f70deab097a",
            "desc": "Traffic light classifier parameter file",
        },
    ],
    "traffic_light_fine_detector": [
        {
            "file": "tlr_car_ped_yolox_s_batch_1.onnx",
            "url": "https://awf.ml.dev.web.auto/perception/models/tlr_yolox_s/v3/tlr_car_ped_yolox_s_batch_1.onnx",
            "sha256": "1ad633066a1195006f4709f8fa07800dd65a74a814b3efb4c99bcc5a1a7962f6",
            "desc": "Traffic light fine detector YOLOX-S batch=1",
        },
        {
            "file": "tlr_labels.txt",
            "url": "https://awf.ml.dev.web.auto/perception/models/tlr_yolox_s/v3/tlr_labels.txt",
            "sha256": "a2a91f5fe9c2e68e3e3647a272bb9bb25cd07631a1990a3fb15efddce7691131",
            "desc": "Fine detector labels",
        },
    ],
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_progress_hook(filename: str):
    """urllib reporthook that prints an inline progress bar."""
    def hook(count: int, block_size: int, total_size: int) -> None:
        downloaded = min(count * block_size, total_size) if total_size > 0 else count * block_size
        if total_size > 0:
            pct = downloaded / total_size * 100
            filled = int(30 * downloaded / total_size)
            bar = "█" * filled + "░" * (30 - filled)
            mb_done = downloaded / 1_048_576
            mb_total = total_size / 1_048_576
            print(f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB  {filename}",
                  end="", flush=True)
        else:
            print(f"\r  {downloaded // 1_048_576} MB  {filename}", end="", flush=True)
    return hook


def download_file(url: str, dest: Path, expected_sha256: str) -> str:
    """
    Download url → dest, verify SHA-256.
    Returns one of: "ok" (already valid), "downloaded", "failed".
    """
    if dest.exists():
        if sha256_of(dest) == expected_sha256:
            print(f"  ✅ already ok      {dest.name}")
            return "ok"
        print(f"  ⚠️  checksum mismatch, re-downloading  {dest.name}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_make_progress_hook(dest.name))
        print()  # newline after progress bar
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"\n  ❌ download error: {exc}")
        return "failed"

    actual = sha256_of(tmp)
    if actual != expected_sha256:
        tmp.unlink()
        print(f"  ❌ checksum mismatch after download")
        print(f"     expected : {expected_sha256}")
        print(f"     got      : {actual}")
        return "failed"

    tmp.rename(dest)
    print(f"  ✅ saved  {dest}")
    return "downloaded"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_list() -> None:
    print("Available model groups:\n")
    for group, files in MODELS.items():
        print(f"  {group}/")
        for entry in files:
            print(f"    {entry['file']:<65}  {entry['desc']}")
    print()


def cmd_download(model_root: Path, groups: list[str]) -> None:
    print(f"Model root : {model_root}\n")

    counts = {"downloaded": 0, "ok": 0, "failed": 0}
    for group in groups:
        print(f"── {group}/")
        for entry in MODELS[group]:
            dest = model_root / group / entry["file"]
            try:
                result = download_file(entry["url"], dest, entry["sha256"])
            except KeyboardInterrupt:
                print("\nInterrupted.")
                sys.exit(1)
            counts[result] = counts.get(result, 0) + 1
        print()

    print(
        f"Done.  "
        f"{counts['downloaded']} downloaded  "
        f"{counts['ok']} already up-to-date  "
        f"{counts['failed']} failed"
    )
    if counts["failed"]:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-root",
        default=os.environ.get("AUTOWARE_MODEL_ROOT", "/opt/autoware/mlmodels"),
        help="Destination root directory (env: AUTOWARE_MODEL_ROOT, default: /opt/autoware/mlmodels)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        metavar="GROUP",
        help=f"Groups to download: {', '.join(MODELS)}. Omit to download all.",
    )
    parser.add_argument("--list", action="store_true", help="List models and exit")
    args = parser.parse_args()

    if args.list:
        cmd_list()
        return

    groups = args.models or list(MODELS.keys())
    unknown = set(groups) - set(MODELS.keys())
    if unknown:
        print(f"Unknown group(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(MODELS.keys())}")
        sys.exit(1)

    cmd_download(Path(args.model_root), groups)


if __name__ == "__main__":
    main()
