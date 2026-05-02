import numpy as np
import cv2
import onnxruntime as ort

from ..base import ModelRunner


class YoloxRunner(ModelRunner):
    """
    Runner for Autoware YOLOX 2D object detection models.

    Expected config keys:
      onnx   - path to .onnx
      label  - path to label.txt (one class name per line)
      params:
        input_size : int, default 960
        strides    : list[int], default [8, 16, 32]
    """

    def load(self, config: dict) -> None:
        self.config = config
        p = config.get("params", {})
        self.input_size: int = p.get("input_size", 960)
        strides: list = p.get("strides", [8, 16, 32])

        self.session = ort.InferenceSession(config["onnx"])
        self.input_name: str = self.session.get_inputs()[0].name

        with open(config["label"]) as f:
            self.class_names: list[str] = [line.strip() for line in f]

        self._grids, self._strides = self._build_grids(strides)

    def _build_grids(self, strides: list) -> tuple[np.ndarray, np.ndarray]:
        grids, stride_vals = [], []
        for s in strides:
            h = w = self.input_size // s
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
            grids.append(grid)
            stride_vals.append(np.full((len(grid), 1), s))
        return (
            np.concatenate(grids, axis=0).astype(np.float32),
            np.concatenate(stride_vals, axis=0).astype(np.float32),
        )

    def preprocess(self, image_bgr: np.ndarray) -> dict:
        h0, w0 = image_bgr.shape[:2]
        scale = min(self.input_size / w0, self.input_size / h0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(image_bgr, (new_w, new_h))

        top = (self.input_size - new_h) // 2
        left = (self.input_size - new_w) // 2
        padded = cv2.copyMakeBorder(
            resized,
            top, self.input_size - new_h - top,
            left, self.input_size - new_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        # BGR->RGB, HWC->CHW, add batch dim; no /255 normalization (Autoware convention)
        tensor = padded[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None]
        return {
            "tensor": tensor,
            "scale": scale,
            "pad_left": left,
            "pad_top": top,
            "orig_w": w0,
            "orig_h": h0,
        }

    def infer(self, preprocess_result: dict) -> dict:
        outputs = self.session.run(None, {self.input_name: preprocess_result["tensor"]})
        return {**preprocess_result, "outputs": outputs}

    def postprocess(self, infer_result: dict, conf_th: float = 0.3, nms_th: float = 0.45) -> list:
        raw = infer_result["outputs"][0][0]  # (N_anchors, 5 + n_classes)
        scale = infer_result["scale"]
        pad_left = infer_result["pad_left"]
        pad_top = infer_result["pad_top"]
        orig_w = infer_result["orig_w"]
        orig_h = infer_result["orig_h"]

        boxes = raw[:, :4].copy()
        obj = raw[:, 4:5]        # objectness (sigmoid already applied by model)
        cls_scores = raw[:, 5:]  # class scores (sigmoid already applied by model)

        # decode center-wh to xyxy in padded-image coords
        boxes[:, :2] = (boxes[:, :2] + self._grids) * self._strides
        boxes[:, 2:4] = np.exp(boxes[:, 2:4]) * self._strides

        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        class_ids = np.argmax(cls_scores, axis=1)
        scores = obj[:, 0] * cls_scores[np.arange(len(cls_scores)), class_ids]

        # map back to original image coords
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_left) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_top) / scale
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, orig_w - 1)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, orig_h - 1)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0, orig_w - 1)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0, orig_h - 1)

        boxes_nms = [
            [int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])]
            for b in xyxy
        ]
        indices = cv2.dnn.NMSBoxes(boxes_nms, scores.tolist(), conf_th, nms_th)
        if len(indices) == 0:
            return []

        return [
            {
                "box": xyxy[i].tolist(),
                "score": float(scores[i]),
                "class_id": int(class_ids[i]),
            }
            for i in indices.flatten()
        ]

    def visualize(
        self,
        image_rgb: np.ndarray,
        results: list,
        colors: dict | None = None,
        thickness: int = 2,
    ) -> np.ndarray:
        img = image_rgb.copy()
        for det in results:
            x1, y1, x2, y2 = map(int, det["box"])
            cls_id = det["class_id"]
            color = (colors or {}).get(cls_id, (0, 255, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            font_scale = 0.4 + thickness * 0.05
            cv2.putText(
                img,
                f"{self.class_names[cls_id]} {det['score']:.2f}",
                (x1, max(y1 - 4, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                max(1, thickness // 2),
            )
        return img
