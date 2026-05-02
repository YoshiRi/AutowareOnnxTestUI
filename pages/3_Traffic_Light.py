import os
import sys
import time

import cv2
import numpy as np
import streamlit as st
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.image.traffic_light import TrafficLightClassifierRunner
from utils.input_source import render_input_source

st.set_page_config(page_title="Traffic Light", layout="wide")
st.title("Traffic Light Classifier")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REGISTRY_PATH = os.path.join(_ROOT, "config", "model_registry.yaml")


@st.cache_data
def _load_registry() -> dict:
    with open(_REGISTRY_PATH) as f:
        return yaml.safe_load(f)


registry = _load_registry()
default_cfg = registry["image"]["traffic_light_classifier"]

st.sidebar.title("Model Config")
onnx_path  = st.sidebar.text_input("ONNX Path",  default_cfg.get("onnx", ""))
label_path = st.sidebar.text_input("Label Path", default_cfg.get("label", ""))

st.sidebar.markdown("""
---
**入力について**

信号機領域のみをクロップした画像を使用してください。
Screen Capture の領域選択機能を使うと、画面上の信号機領域を直接切り出せます。
""")


@st.cache_resource
def _load_model(onnx: str, label: str, params: dict) -> TrafficLightClassifierRunner:
    runner = TrafficLightClassifierRunner()
    runner.load({
        "onnx": onnx,
        "label": label if os.path.exists(label) else None,
        "params": params,
    })
    return runner


if not os.path.exists(onnx_path):
    st.warning(f"ONNX model not found: `{onnx_path}`")
    st.info("Set the correct path in the sidebar or update `config/model_registry.yaml`.")
    st.stop()

runner = _load_model(onnx_path, label_path, default_cfg.get("params", {}))

# --- Input source ---
image_bgr = render_input_source(key="tl")

if image_bgr is not None:
    with st.spinner("Running inference…"):
        pre = runner.preprocess(image_bgr)
        t0  = time.perf_counter()
        inf = runner.infer(pre)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    st.session_state["tl_inf"] = {
        "inf": inf,
        "image_rgb": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        "elapsed_ms": elapsed_ms,
    }

if cached := st.session_state.get("tl_inf"):
    results    = runner.postprocess(cached["inf"])
    annotated  = runner.visualize(cached["image_rgb"], results)
    result     = results[0]
    image_rgb  = cached["image_rgb"]
    elapsed_ms = cached["elapsed_ms"]

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Input", use_container_width=True)
    with col2:
        st.image(annotated, caption=f"Result — {elapsed_ms:.1f} ms", use_container_width=True)

    st.markdown("### Class Probabilities")
    for i, (name, prob) in enumerate(zip(result["class_names"], result["probs"])):
        st.progress(float(prob), text=f"{name}: {prob:.3f}")
