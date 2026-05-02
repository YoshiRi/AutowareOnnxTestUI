import os
import sys
import time

import cv2
import numpy as np
import streamlit as st
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.image.semantic_seg import SemanticSegRunner

st.set_page_config(page_title="Segmentation", layout="wide")
st.title("Semantic Segmentation")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REGISTRY_PATH = os.path.join(_ROOT, "config", "model_registry.yaml")


@st.cache_data
def _load_registry() -> dict:
    with open(_REGISTRY_PATH) as f:
        return yaml.safe_load(f)


registry = _load_registry()
default_cfg = registry["image"]["semantic_seg"]

# resolve relative color_map path against repo root
_raw_color_map = default_cfg.get("color_map", "")
_default_color_map = (
    os.path.join(_ROOT, _raw_color_map)
    if _raw_color_map and not os.path.isabs(_raw_color_map)
    else _raw_color_map
)

# --- Sidebar ---
st.sidebar.title("Model Config")
onnx_path = st.sidebar.text_input("ONNX Path", default_cfg.get("onnx", ""))
color_map_path = st.sidebar.text_input("Color Map CSV", _default_color_map)
alpha = st.sidebar.slider("Overlay Alpha", 0.0, 1.0, 0.5, 0.05)


@st.cache_resource
def _load_model(onnx: str, color_map: str, params: dict) -> SemanticSegRunner:
    runner = SemanticSegRunner()
    runner.load({
        "onnx": onnx,
        "color_map": color_map if os.path.exists(color_map) else None,
        "params": params,
    })
    return runner


if not os.path.exists(onnx_path):
    st.warning(f"ONNX model not found: `{onnx_path}`")
    st.info("Set the correct path in the sidebar or update `config/model_registry.yaml`.")
    st.stop()

runner = _load_model(onnx_path, color_map_path, default_cfg.get("params", {}))

# Class legend
st.sidebar.markdown("---")
st.sidebar.markdown("### Class Legend")
for i, name in enumerate(runner.class_names):
    r, g, b = runner.color_map[i]
    st.sidebar.markdown(
        f"<span style='color:rgb({r},{g},{b})'>■</span> {i}: {name}",
        unsafe_allow_html=True,
    )

# --- Main ---
uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded:
    image_rgb = np.array(Image.open(uploaded).convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    with st.spinner("Running inference…"):
        pre = runner.preprocess(image_bgr)
        t0 = time.perf_counter()
        inf = runner.infer(pre)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results = runner.postprocess(inf)

    overlay = runner.visualize(image_rgb, results, alpha=alpha)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Input", use_container_width=True)
    with col2:
        st.image(overlay, caption=f"Segmentation — {elapsed_ms:.1f} ms", use_container_width=True)
