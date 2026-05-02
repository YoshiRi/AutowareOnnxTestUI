"""
Shared input source widget for Streamlit image pages.

Usage:
    from utils.input_source import render_input_source

    image_bgr = render_input_source(key="det")
    if image_bgr is not None:
        # new image arrived — run ONNX inference and cache the result
        ...

Returns a BGR np.ndarray when a new image is ready, otherwise None.
Callers should cache inference results in st.session_state so results
persist while the user adjusts sliders or the crop region.

Sources:
  File Upload   — st.file_uploader (immediate on upload)
  Camera        — st.camera_input  (immediate on capture)
  Screen Capture — mss screenshot + streamlit-cropper region selector;
                   returns image only when the user clicks ▶ Run
"""

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image


def render_input_source(key: str = "img") -> np.ndarray | None:
    """Render source selector and return BGR image when ready, else None."""
    source = st.radio(
        "Input source",
        ["File Upload", "Camera", "Screen Capture"],
        horizontal=True,
        key=f"{key}_source",
    )

    if source == "File Upload":
        return _file_upload(key)
    if source == "Camera":
        return _camera(key)
    return _screen_capture(key)


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

def _file_upload(key: str) -> np.ndarray | None:
    uploaded = st.file_uploader(
        "Upload image", type=["png", "jpg", "jpeg"], key=f"{key}_upload"
    )
    if uploaded is None:
        return None
    return _pil_to_bgr(Image.open(uploaded).convert("RGB"))


# ---------------------------------------------------------------------------
# Camera snapshot
# ---------------------------------------------------------------------------

def _camera(key: str) -> np.ndarray | None:
    frame = st.camera_input("Take a photo", key=f"{key}_camera")
    if frame is None:
        return None
    return _pil_to_bgr(Image.open(frame).convert("RGB"))


# ---------------------------------------------------------------------------
# Screen capture
# ---------------------------------------------------------------------------

def _screen_capture(key: str) -> np.ndarray | None:
    try:
        import mss
    except ImportError:
        st.error("`mss` が見つかりません。`uv sync` を実行してください。")
        return None

    # Monitor selector + Capture button on the same row
    with mss.mss() as sct:
        monitors = sct.monitors[1:]  # index 0 = all monitors combined

    if not monitors:
        st.error("モニターが検出されませんでした。")
        return None

    col_sel, col_btn = st.columns([4, 1])
    with col_sel:
        mon_idx = st.selectbox(
            "Monitor",
            range(len(monitors)),
            format_func=lambda i: f"Monitor {i + 1}  ({monitors[i]['width']} × {monitors[i]['height']})",
            key=f"{key}_monitor",
        )
    with col_btn:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        do_capture = st.button("📸 Capture", key=f"{key}_capture", use_container_width=True)

    if do_capture:
        with mss.mss() as sct:
            raw = np.array(sct.grab(sct.monitors[mon_idx + 1]))  # BGRA
        # BGRA → RGB for PIL storage
        rgb = raw[:, :, [2, 1, 0]]
        st.session_state[f"{key}_screenshot"] = Image.fromarray(rgb)

    screenshot: Image.Image | None = st.session_state.get(f"{key}_screenshot")
    if screenshot is None:
        st.info("📸 Capture ボタンで画面を取得し、推論したい領域を選択してください。")
        return None

    st.markdown("**領域を選択**（緑のボックスをドラッグ・リサイズ）")
    cropped = _region_selector(screenshot, key)

    # Preview of the current selection
    if cropped is not None and cropped.size[0] > 0 and cropped.size[1] > 0:
        st.image(cropped, caption=f"選択領域プレビュー  {cropped.width} × {cropped.height} px", width=320)

    if st.button("▶ Run on Selection", key=f"{key}_run", type="primary"):
        if cropped is None or cropped.size[0] == 0:
            st.warning("有効な領域を選択してください。")
            return None
        return _pil_to_bgr(cropped.convert("RGB"))

    return None


def _region_selector(screenshot: Image.Image, key: str) -> Image.Image | None:
    """
    Show the screenshot with an interactive crop box.
    Falls back to number_input sliders if streamlit-cropper is unavailable.
    """
    try:
        from streamlit_cropper import st_cropper

        cropped: Image.Image = st_cropper(
            screenshot,
            realtime_update=True,
            box_color="#00FF00",
            key=f"{key}_cropper",
        )
        return cropped

    except ImportError:
        pass

    # Slider fallback -------------------------------------------------------
    w, h = screenshot.size
    c1, c2, c3, c4 = st.columns(4)
    left   = int(c1.number_input("Left",   0, w - 1, 0,     step=1, key=f"{key}_left"))
    top    = int(c2.number_input("Top",    0, h - 1, 0,     step=1, key=f"{key}_top"))
    right  = int(c3.number_input("Right",  1, w,     w,     step=1, key=f"{key}_right"))
    bottom = int(c4.number_input("Bottom", 1, h,     h,     step=1, key=f"{key}_bottom"))

    # Draw the selection rectangle on a downscaled preview
    preview = np.array(screenshot.copy())
    cv2.rectangle(preview, (left, top), (right, bottom), (0, 255, 0), max(2, h // 200))
    st.image(preview, caption="プレビュー（緑 = 選択領域）", use_container_width=True)

    if right <= left or bottom <= top:
        st.warning("Right > Left、Bottom > Top になるように設定してください。")
        return None
    return screenshot.crop((left, top, right, bottom))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
