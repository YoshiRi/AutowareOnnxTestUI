import os
import sys
import time

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.image.semantic_seg import SemanticSegRunner
from utils.config_loader import (
    load_registry,
    render_model_root_sidebar,
    render_resolved_paths_expander,
    resolve_model_config,
)
from utils.input_source import render_input_source

st.set_page_config(page_title="Segmentation", layout="wide")
st.title("Semantic Segmentation")

registry = load_registry()

# --- Sidebar ---
st.sidebar.title("Model Config")
model_root = render_model_root_sidebar(registry)
cfg = resolve_model_config(registry["image"]["semantic_seg"], model_root)
render_resolved_paths_expander(cfg)

st.sidebar.markdown("---")
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


if not os.path.exists(cfg["onnx"]):
    st.warning(f"ONNX model not found: `{cfg['onnx']}`")
    st.info("Set **Model Root** in the sidebar, or update `config/model_registry.yaml`.")
    st.stop()

runner = _load_model(cfg["onnx"], cfg.get("color_map", ""), cfg.get("params", {}))

st.sidebar.markdown("---")
st.sidebar.markdown("### Class Legend")
for i, name in enumerate(runner.class_names):
    r, g, b = runner.color_map[i]
    st.sidebar.markdown(
        f"<span style='color:rgb({r},{g},{b})'>■</span> {i}: {name}",
        unsafe_allow_html=True,
    )

# --- Input source ---
image_bgr = render_input_source(key="seg")

if image_bgr is not None:
    with st.spinner("Running inference…"):
        pre = runner.preprocess(image_bgr)
        t0  = time.perf_counter()
        inf = runner.infer(pre)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    st.session_state["seg_inf"] = {
        "inf": inf,
        "image_rgb": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        "elapsed_ms": elapsed_ms,
    }

# Re-run postprocess/visualize — alpha slider updates instantly
if cached := st.session_state.get("seg_inf"):
    results    = runner.postprocess(cached["inf"])
    overlay    = runner.visualize(cached["image_rgb"], results, alpha=alpha)
    image_rgb  = cached["image_rgb"]
    elapsed_ms = cached["elapsed_ms"]

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Input", use_container_width=True)
    with col2:
        st.image(overlay, caption=f"Segmentation — {elapsed_ms:.1f} ms", use_container_width=True)
