import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
import random
import time
import matplotlib.pyplot as plt
from collections import defaultdict

INPUT_SIZE = 960

# -----------------------------
# モデルフォルダ選択
# -----------------------------
st.sidebar.title("Model Selection")

model_root = st.sidebar.text_input(
    "Model Folder",
    "/opt/autoware/mlmodels/yolox"
)

def find_model_files(folder):
    onnx_file = None
    label_file = None
    for f in os.listdir(folder):
        if f.endswith(".onnx"):
            onnx_file = os.path.join(folder, f)
        if f == "label.txt":
            label_file = os.path.join(folder, f)
    return onnx_file, label_file


onnx_path, label_path = find_model_files(model_root)

if not onnx_path or not label_path:
    st.error("onnx または label.txt が見つかりません")
    st.stop()

# -----------------------------
# モデル読み込み（キャッシュ）
# -----------------------------
@st.cache_resource
def load_model(path):
    session = ort.InferenceSession(path)
    return session

session = load_model(onnx_path)
input_name = session.get_inputs()[0].name

with open(label_path) as f:
    class_names = [l.strip() for l in f.readlines()]

# -----------------------------
# 色生成（安定）
# -----------------------------
def generate_colors(num_classes):
    random.seed(42)
    return [
        tuple(random.randint(50,255) for _ in range(3))
        for _ in range(num_classes)
    ]

colors = generate_colors(len(class_names))

# -----------------------------
# 前処理（/255無し）
# -----------------------------
def preprocess(image):
    h0, w0 = image.shape[:2]
    scale = min(INPUT_SIZE / w0, INPUT_SIZE / h0)

    new_w, new_h = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(image, (new_w, new_h))

    top = (INPUT_SIZE - new_h) // 2
    left = (INPUT_SIZE - new_w) // 2

    padded = cv2.copyMakeBorder(
        resized,
        top, INPUT_SIZE-new_h-top,
        left, INPUT_SIZE-new_w-left,
        cv2.BORDER_CONSTANT,
        value=(114,114,114)
    )

    padded = padded.astype(np.float32)  # ← /255しない
    padded = np.transpose(padded, (2,0,1))
    padded = np.expand_dims(padded, axis=0)

    return padded, scale, left, top, w0, h0

# -----------------------------
# Grid生成
# -----------------------------
def create_grids():
    strides = [8,16,32]
    grids = []
    strides_list = []
    for s in strides:
        h = INPUT_SIZE//s
        w = INPUT_SIZE//s
        xv,yv = np.meshgrid(np.arange(w),np.arange(h))
        grid = np.stack((xv,yv),2).reshape(-1,2)
        grids.append(grid)
        strides_list.append(np.full((grid.shape[0],1),s))
    return np.concatenate(grids), np.concatenate(strides_list)

grids, expanded_strides = create_grids()

# -----------------------------
# 後処理
# -----------------------------
def postprocess(outputs, scale, pad_left, pad_top, orig_w, orig_h, nms_th):

    raw = outputs[0][0]  # (18900,13)

    boxes = raw[:,:4]
    obj = raw[:,4:5]     # ← sigmoid済み
    cls_scores = raw[:,5:]  # ← sigmoid済み

    boxes[:,:2] = (boxes[:,:2] + grids) * expanded_strides
    boxes[:,2:4] = np.exp(boxes[:,2:4]) * expanded_strides

    xyxy = np.zeros_like(boxes)
    xyxy[:,0] = boxes[:,0] - boxes[:,2]/2
    xyxy[:,1] = boxes[:,1] - boxes[:,3]/2
    xyxy[:,2] = boxes[:,0] + boxes[:,2]/2
    xyxy[:,3] = boxes[:,1] + boxes[:,3]/2

    class_ids = np.argmax(cls_scores,1)
    scores = obj[:,0] * cls_scores[np.arange(len(cls_scores)), class_ids]

    if len(xyxy)==0:
        return []

    xyxy[:,[0,2]] = (xyxy[:,[0,2]] - pad_left) / scale
    xyxy[:,[1,3]] = (xyxy[:,[1,3]] - pad_top) / scale

    xyxy[:,0] = np.clip(xyxy[:,0],0,orig_w-1)
    xyxy[:,1] = np.clip(xyxy[:,1],0,orig_h-1)
    xyxy[:,2] = np.clip(xyxy[:,2],0,orig_w-1)
    xyxy[:,3] = np.clip(xyxy[:,3],0,orig_h-1)

    boxes_for_nms = [
        [int(b[0]),int(b[1]),int(b[2]-b[0]),int(b[3]-b[1])]
        for b in xyxy
    ]

    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms,
        scores.tolist(),
        score_threshold=0.0,
        nms_threshold=nms_th
    )

    if len(indices)==0:
        return []

    return [
        (xyxy[i], scores[i], class_ids[i])
        for i in indices.flatten()
    ]

# -----------------------------
# UI
# -----------------------------
FIXED_COLORS = {
    0: (0, 255, 0),      
    1: (0, 0, 255),      
    2: (255, 0, 0),      
    3: (0, 255, 255),
    4: (255, 0, 255),
    5: (255, 255, 0),
    6: (128, 128, 255),
    7: (255, 128, 0),
}

st.title("YOLOX Model Validator")

conf_th = st.sidebar.slider("Confidence",0.0,1.0,0.3,0.01)
nms_th = st.sidebar.slider("NMS",0.0,1.0,0.45,0.01)

st.sidebar.markdown("### Class Legend")
for i,name in enumerate(class_names):
    color = FIXED_COLORS.get(i, (255,255,255))
    st.sidebar.markdown(
        f"<span style='color:rgb{color}'>■ {i}: {name}</span>",
        unsafe_allow_html=True
    )

box_thickness = st.sidebar.slider(
    "Box Thickness",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

uploaded = st.file_uploader("Upload Image",type=["png","jpg","jpeg"])

if uploaded:
    image = np.array(Image.open(uploaded))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    input_tensor, scale, pad_left, pad_top, w0, h0 = preprocess(image_bgr)

    start = time.time()
    outputs = session.run(None,{input_name:input_tensor})
    detections = postprocess(outputs,scale,pad_left,pad_top,w0,h0,nms_th)
    end = time.time()

    all_detections = detections  # postprocessの返り値
    filtered_detections = [
        (box, score, cls)
        for (box, score, cls) in all_detections
        if score > conf_th
    ]

    for box,score,cls in filtered_detections:
        x1,y1,x2,y2 = box
        color = FIXED_COLORS.get(cls, (255,255,255))
        cv2.rectangle(
            image,
            (int(x1),int(y1)),
            (int(x2),int(y2)),
            color,
            box_thickness
        )
        font_scale = 0.5 + box_thickness * 0.1
        text_thickness = max(1, box_thickness // 2)

        cv2.putText(
            image,
            f"{class_names[cls]} {score:.2f}",
            (int(x1), int(y1)-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            text_thickness
        )

    st.image(image,caption=f"Inference {(end-start)*1000:.1f} ms")

    if len(all_detections) > 0:
        scores = [d[1] for d in all_detections]
        class_score_map = defaultdict(list)
        for _, score, cls in all_detections:
            class_score_map[cls].append(score)

        bins = np.linspace(0, 1, 30)
        fig, ax = plt.subplots()

        plotted = False
        for cls, class_scores in class_score_map.items():
            if len(class_scores) < 2:
                continue

            hist, bin_edges = np.histogram(class_scores, bins=bins)
            if hist.sum() == 0:
                continue
            hist = hist / hist.sum()
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            ax.plot(
                centers,
                hist,
                label=class_names[cls],
            )
            plotted = True

        if plotted:
            ax.axvline(conf_th, linestyle="--")
            ax.set_title("Score Distribution (After NMS)")
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Normalized Frequency")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("NMS後のスコア分布を描画するには、各クラスで2件以上の検出が必要です。")

        plt.close(fig)