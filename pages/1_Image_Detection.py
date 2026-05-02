import os
import sys
import time
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.image.yolox import YoloxRunner
from utils.visualization import class_color_map, sidebar_class_legend

st.set_page_config(page_title="Image Detection", layout="wide")
st.title("Image Detection — YOLOX")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REGISTRY_PATH = os.path.join(_ROOT, "config", "model_registry.yaml")


@st.cache_data
def _load_registry() -> dict:
    with open(_REGISTRY_PATH) as f:
        return yaml.safe_load(f)


registry = _load_registry()
default_cfg = registry["image"]["yolox"]

# --- Sidebar: model paths ---
st.sidebar.title("Model Config")
onnx_path = st.sidebar.text_input("ONNX Path", default_cfg.get("onnx", ""))
label_path = st.sidebar.text_input("Label Path", default_cfg.get("label", ""))

# --- Sidebar: inference params ---
st.sidebar.markdown("---")
conf_th = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.01)
nms_th = st.sidebar.slider("NMS Threshold", 0.0, 1.0, 0.45, 0.01)
box_thickness = st.sidebar.slider("Box Thickness", 1, 10, 2)


@st.cache_resource
def _load_model(onnx: str, label: str, params: dict) -> YoloxRunner:
    runner = YoloxRunner()
    runner.load({"onnx": onnx, "label": label, "params": params})
    return runner


# Guard: check files exist before loading
if not os.path.exists(onnx_path):
    st.warning(f"ONNX model not found: `{onnx_path}`")
    st.info("Set the correct path in the sidebar or update `config/model_registry.yaml`.")
    st.stop()

if not os.path.exists(label_path):
    st.warning(f"Label file not found: `{label_path}`")
    st.stop()

runner = _load_model(onnx_path, label_path, default_cfg.get("params", {}))
colors = class_color_map(len(runner.class_names))
sidebar_class_legend(st, runner.class_names, colors)

# --- Main: upload and infer ---
uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded:
    image_rgb = np.array(Image.open(uploaded).convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    with st.spinner("Running inference…"):
        pre = runner.preprocess(image_bgr)
        t0 = time.perf_counter()
        inf = runner.infer(pre)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results = runner.postprocess(inf, conf_th=conf_th, nms_th=nms_th)

    annotated = runner.visualize(image_rgb, results, colors=colors, thickness=box_thickness)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Input", use_container_width=True)
    with col2:
        st.image(
            annotated,
            caption=f"{len(results)} detections — {elapsed_ms:.1f} ms",
            use_container_width=True,
        )

    # Score distribution chart (post-NMS)
    if results:
        class_score_map: dict = defaultdict(list)
        for det in results:
            class_score_map[det["class_id"]].append(det["score"])

        bins = np.linspace(0, 1, 30)
        fig, ax = plt.subplots(figsize=(7, 3))
        plotted = False
        for cls_id, cls_scores in class_score_map.items():
            if len(cls_scores) < 2:
                continue
            hist, edges = np.histogram(cls_scores, bins=bins)
            if hist.sum() == 0:
                continue
            centers = (edges[:-1] + edges[1:]) / 2
            r, g, b = colors.get(cls_id, (200, 200, 200))
            ax.plot(centers, hist / hist.sum(), label=runner.class_names[cls_id],
                    color=(r / 255, g / 255, b / 255))
            plotted = True

        if plotted:
            ax.axvline(conf_th, color="gray", linestyle="--", label="conf threshold")
            ax.set_title("Score Distribution (post-NMS, per class)")
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Normalized Frequency")
            ax.legend(fontsize=8)
            st.pyplot(fig)
        plt.close(fig)
