#!/usr/bin/env python3
"""
Smoke-test the inference pipeline for all available models.

For each model defined in model_registry.yaml the script:
  1. Locates the model directory under model_root
  2. Finds an ONNX file (picks the first alphabetically)
  3. Loads the runner, preprocesses a sample image, runs infer + postprocess
  4. Reports ✅ PASS / ⚠️ SKIP (no ONNX found) / ❌ FAIL

Usage:
  uv run python scripts/test_pipeline.py
  uv run python scripts/test_pipeline.py --model-root ~/autoware_models
  uv run python scripts/test_pipeline.py --sample bus.jpg   # override sample image
"""

import argparse
import glob
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import yaml

_REPO_ROOT = Path(__file__).parent.parent
_REGISTRY_PATH = _REPO_ROOT / "config" / "model_registry.yaml"
_SAMPLES_DIR = _REPO_ROOT / "samples"

sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_onnx(model_dir: str) -> str | None:
    files = sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
    return files[0] if files else None


def _resolve(base: str, path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.join(base, path)


def _load_sample_image(sample_path: str | None) -> np.ndarray | None:
    """Return BGR image from samples dir, or a synthetic grey image."""
    if sample_path and os.path.isfile(sample_path):
        img = cv2.imread(sample_path)
        if img is not None:
            return img

    # Try any file in samples/
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        found = sorted(glob.glob(str(_SAMPLES_DIR / ext)))
        if found:
            img = cv2.imread(found[0])
            if img is not None:
                print(f"  Using sample image: {found[0]}")
                return img

    # Fallback: synthetic 960×960 grey image
    print("  ⚠️  No sample image found — using synthetic 960×960 grey image")
    print(f"     Run: uv run python scripts/download_models.py --samples")
    return np.full((960, 960, 3), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Per-model runners
# ---------------------------------------------------------------------------

def _test_yolox(cfg: dict, onnx: str, image_bgr: np.ndarray) -> None:
    from models.image.yolox import YoloxRunner
    runner = YoloxRunner()
    runner.load({"onnx": onnx, "label": cfg.get("label"), "params": cfg.get("params", {})})
    pre = runner.preprocess(image_bgr)
    inf = runner.infer(pre)
    results = runner.postprocess(inf)
    _ = runner.visualize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), results)
    print(f"  detections: {len(results)}")


def _test_semantic_seg(cfg: dict, onnx: str, image_bgr: np.ndarray) -> None:
    from models.image.semantic_seg import SemanticSegRunner
    runner = SemanticSegRunner()
    runner.load({
        "onnx": onnx,
        "color_map": cfg.get("color_map"),
        "params": cfg.get("params", {}),
    })
    pre = runner.preprocess(image_bgr)
    inf = runner.infer(pre)
    results = runner.postprocess(inf)
    _ = runner.visualize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), results)
    print(f"  classes: {len(runner.class_names)}")


def _test_traffic_light_classifier(cfg: dict, onnx: str, image_bgr: np.ndarray) -> None:
    from models.image.traffic_light import TrafficLightClassifierRunner
    runner = TrafficLightClassifierRunner()
    runner.load({
        "onnx": onnx,
        "label": cfg.get("label"),
        "params": cfg.get("params", {}),
    })
    # Resize to expected input size
    params = cfg.get("params", {})
    sz = params.get("input_size", [224, 224])
    if isinstance(sz, int):
        sz = [sz, sz]
    crop = cv2.resize(image_bgr, (sz[1], sz[0]))
    pre = runner.preprocess(crop)
    inf = runner.infer(pre)
    results = runner.postprocess(inf)
    _ = runner.visualize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), results)
    print(f"  top class: {results[0]['class_names'][int(np.argmax(results[0]['probs']))]}")


def _test_traffic_light_fine_detector(cfg: dict, onnx: str, image_bgr: np.ndarray) -> None:
    from models.image.yolox import YoloxRunner
    runner = YoloxRunner()
    runner.load({"onnx": onnx, "label": cfg.get("label"), "params": cfg.get("params", {})})
    pre = runner.preprocess(image_bgr)
    inf = runner.infer(pre)
    results = runner.postprocess(inf)
    _ = runner.visualize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), results)
    print(f"  detections: {len(results)}")


_RUNNERS: dict[str, dict] = {
    "yolox": {
        "fn": _test_yolox,
        "category": "image",
        "desc": "YOLOX object detection",
    },
    "semantic_seg": {
        "fn": _test_semantic_seg,
        "category": "image",
        "desc": "Semantic segmentation",
    },
    "traffic_light_classifier": {
        "fn": _test_traffic_light_classifier,
        "category": "image",
        "desc": "Traffic light classifier",
    },
    "traffic_light_fine_detector": {
        "fn": _test_traffic_light_fine_detector,
        "category": "image",
        "desc": "Traffic light fine detector",
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-root",
        default=os.environ.get("AUTOWARE_MODEL_ROOT", "/opt/autoware/mlmodels"),
        help="Root directory for models (env: AUTOWARE_MODEL_ROOT)",
    )
    parser.add_argument(
        "--sample",
        metavar="PATH",
        help="Path to a sample image (default: first file in samples/)",
    )
    args = parser.parse_args()

    with open(_REGISTRY_PATH) as f:
        registry = yaml.safe_load(f)

    model_root = args.model_root
    print(f"Model root : {model_root}")

    image_bgr = _load_sample_image(args.sample)
    print(f"Image shape: {image_bgr.shape}\n")

    counts = {"pass": 0, "skip": 0, "fail": 0}

    for model_key, info in _RUNNERS.items():
        category = info["category"]
        desc = info["desc"]
        cfg_raw = registry.get(category, {}).get(model_key, {})
        if not cfg_raw:
            continue

        # Resolve paths
        cfg = dict(cfg_raw)
        for k in ("model_dir", "label", "color_map"):
            if k in cfg:
                cfg[k] = _resolve(model_root, cfg[k])
        if "param_files" in cfg:
            cfg["param_files"] = {k: _resolve(model_root, v) for k, v in cfg["param_files"].items()}

        model_dir = cfg.get("model_dir", "")
        onnx = _find_onnx(model_dir) if model_dir else None

        label = f"{model_key} ({desc})"
        if onnx is None:
            print(f"⚠️  SKIP  {label}")
            if model_dir:
                print(f"         No .onnx in: {model_dir}")
            counts["skip"] += 1
            continue

        print(f"▶  TEST   {label}")
        print(f"   onnx : {os.path.basename(onnx)}")
        try:
            info["fn"](cfg, onnx, image_bgr.copy())
            print(f"✅ PASS   {label}\n")
            counts["pass"] += 1
        except Exception:
            print(f"❌ FAIL   {label}")
            traceback.print_exc()
            print()
            counts["fail"] += 1

    print("─" * 60)
    print(f"Results: {counts['pass']} passed  {counts['skip']} skipped  {counts['fail']} failed")

    if counts["fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
