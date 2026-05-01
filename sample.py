import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import time

MODEL_PATH = "/opt/autoware/mlmodels/yolox/yolox-sPlus-T4-960x960-pseudo-finetune.onnx"
LABEL_PATH = "/opt/autoware/mlmodels/yolox/label.txt"
INPUT_SIZE = 960

# -----------------------
# Load labels
# -----------------------
with open(LABEL_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# -----------------------
# Load model (cached)
# -----------------------
@st.cache_resource
def load_model():
    session = ort.InferenceSession(MODEL_PATH)
    return session

session = load_model()
input_name = session.get_inputs()[0].name

# -----------------------
# Grid生成
# -----------------------
@st.cache_resource
def create_grids():
    strides = [8, 16, 32]
    grids = []
    expanded_strides = []
    for stride in strides:
        h = INPUT_SIZE // stride
        w = INPUT_SIZE // stride
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((grid.shape[0], 1), stride))
    return (
        np.concatenate(grids, axis=0).astype(np.float32),
        np.concatenate(expanded_strides, axis=0).astype(np.float32),
    )

grids, expanded_strides = create_grids()

# -----------------------
# Preprocess
# -----------------------
def preprocess(image):
    h0, w0 = image.shape[:2]
    scale = min(INPUT_SIZE / w0, INPUT_SIZE / h0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)

    resized = cv2.resize(image, (new_w, new_h))

    top = (INPUT_SIZE - new_h) // 2
    bottom = INPUT_SIZE - new_h - top
    left = (INPUT_SIZE - new_w) // 2
    right = INPUT_SIZE - new_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    padded = padded.astype(np.float32)
    padded = np.transpose(padded, (2, 0, 1))
    padded = np.expand_dims(padded, axis=0)

    return padded, scale, left, top, w0, h0

# -----------------------
# Postprocess
# -----------------------
def postprocess(outputs, scale, pad_left, pad_top, orig_w, orig_h, conf_th, nms_th):

    raw = outputs[0][0]  # (18900, 13)

    print("raw min/max:", raw.min(), raw.max())
    print("raw bbox part min/max:", raw[:, :4].min(), raw[:, :4].max())
    print("raw obj min/max:", raw[:, 4].min(), raw[:, 4].max())

    boxes = raw[:, :4]
    obj = raw[:, 4:5]
    cls_scores = raw[:, 5:]
    print("after sigmoid obj max:", obj.max())
    print("after sigmoid cls max:", cls_scores.max())
    # print("final score max:", scores.max())

    boxes[:, :2] = (boxes[:, :2] + grids) * expanded_strides
    boxes[:, 2:4] = np.exp(boxes[:, 2:4]) * expanded_strides

    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    class_ids = np.argmax(cls_scores, axis=1)
    scores = obj[:, 0] * cls_scores[np.arange(len(cls_scores)), class_ids]

    mask = scores > conf_th
    xyxy = xyxy[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(xyxy) == 0:
        return []

    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_left) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_top) / scale

    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, orig_w - 1)
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, orig_h - 1)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, orig_w - 1)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, orig_h - 1)

    boxes_for_nms = [
        [int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])]
        for b in xyxy
    ]

    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms,
        scores.tolist(),
        conf_th,
        nms_th
    )

    if len(indices) == 0:
        return []

    return [(xyxy[i], scores[i], class_ids[i]) for i in indices.flatten()]

# -----------------------
# UI
# -----------------------
st.title("YOLOX Streamlit Visualizer")

conf_th = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.01)
nms_th = st.sidebar.slider("NMS Threshold", 0.0, 1.0, 0.45, 0.01)

st.sidebar.markdown("### Class Legend")
for i, name in enumerate(class_names):
    st.sidebar.write(f"{i}: {name}")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    input_tensor, scale, pad_left, pad_top, w0, h0 = preprocess(image_bgr)

    start = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    detections = postprocess(outputs, scale, pad_left, pad_top, w0, h0, conf_th, nms_th)
    end = time.time()

    for box, score, cls in detections:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(image, f"{class_names[cls]} {score:.2f}",
                    (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,0), 1)

    st.image(image, caption=f"Inference Time: {(end-start)*1000:.1f} ms")