import numpy as np
import cv2
import onnxruntime as ort

from ..base import ModelRunner

# State -> display color (RGB)
_STATE_COLORS = {
    "GREEN": (0, 200, 0),
    "YELLOW": (220, 200, 0),
    "RED": (200, 0, 0),
    "UNKNOWN": (128, 128, 128),
}


class TrafficLightClassifierRunner(ModelRunner):
    """
    Runner for Autoware traffic-light classifier models.

    Accepts a single cropped traffic-light image and returns the signal state.

    Expected config keys:
      onnx  - path to .onnx
      label - path to label.txt (optional; default: GREEN/YELLOW/RED/UNKNOWN)
      params:
        input_size : [H, W], default [224, 224]
        mean       : [R, G, B], default ImageNet mean
        std        : [R, G, B], default ImageNet std
    """

    def load(self, config: dict) -> None:
        self.config = config
        p = config.get("params", {})
        hw = p.get("input_size", [224, 224])
        if isinstance(hw, int):
            self.input_h, self.input_w = hw, hw
        else:
            self.input_h, self.input_w = int(hw[0]), int(hw[1])

        mean = np.array(p.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.array(p.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        self._mean = mean.reshape(3, 1, 1)
        self._std = std.reshape(3, 1, 1)

        self.session = ort.InferenceSession(config["onnx"])
        self.input_name: str = self.session.get_inputs()[0].name

        label_path = config.get("label")
        if label_path:
            with open(label_path) as f:
                self.class_names: list[str] = [line.strip() for line in f]
        else:
            self.class_names = ["GREEN", "YELLOW", "RED", "UNKNOWN"]

    def preprocess(self, image_bgr: np.ndarray) -> dict:
        h0, w0 = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h))
        tensor = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        tensor = (tensor - self._mean) / self._std
        return {
            "tensor": tensor[None].astype(np.float32),
            "orig_h": h0,
            "orig_w": w0,
        }

    def infer(self, preprocess_result: dict) -> dict:
        outputs = self.session.run(None, {self.input_name: preprocess_result["tensor"]})
        return {**preprocess_result, "outputs": outputs}

    def postprocess(self, infer_result: dict, **params) -> list:
        logits = infer_result["outputs"][0][0]
        # softmax
        shifted = logits - logits.max()
        probs = np.exp(shifted) / np.exp(shifted).sum()
        class_id = int(np.argmax(probs))
        label = self.class_names[class_id] if class_id < len(self.class_names) else "UNKNOWN"
        return [
            {
                "class_id": class_id,
                "label": label,
                "probs": probs.tolist(),
                "class_names": self.class_names,
            }
        ]

    def visualize(
        self,
        image_rgb: np.ndarray,
        results: list,
        **kwargs,
    ) -> np.ndarray:
        result = results[0]
        label = result["label"]
        conf = result["probs"][result["class_id"]]
        color = _STATE_COLORS.get(label.upper(), (128, 128, 128))

        img = image_rgb.copy()
        text = f"{label}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(img, (8, 8), (12 + tw, 12 + th + 4), (0, 0, 0), -1)
        cv2.putText(img, text, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        return img
