import numpy as np
import cv2
import pandas as pd
import onnxruntime as ort

from ..base import ModelRunner


class SemanticSegRunner(ModelRunner):
    """
    Runner for Autoware semantic segmentation models (e.g. ERFNet-based).

    Expected config keys:
      onnx      - path to .onnx
      color_map - path to CSV with columns: id, name, r, g, b (optional)
      params:
        input_size : [H, W], default [512, 512]
        mean       : [R, G, B], default ImageNet mean
        std        : [R, G, B], default ImageNet std
    """

    def load(self, config: dict) -> None:
        self.config = config
        p = config.get("params", {})
        hw = p.get("input_size", [512, 512])
        # input_size can be int (square) or [H, W]
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

        color_map_path = config.get("color_map")
        if color_map_path:
            df = pd.read_csv(color_map_path)
            self.class_names: list[str] = df["name"].tolist()
            self.color_map: np.ndarray = df[["r", "g", "b"]].values.astype(np.uint8)
        else:
            # fallback: derive n_classes from model output shape if available
            out_shape = self.session.get_outputs()[0].shape
            n = out_shape[1] if len(out_shape) == 4 else 20
            rng = np.random.default_rng(42)
            self.color_map = rng.integers(50, 255, size=(n, 3), dtype=np.uint8)
            self.class_names = [str(i) for i in range(n)]

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
        raw = infer_result["outputs"][0]  # (1, C, H, W) or (1, H, W)
        if raw.ndim == 4:
            class_map = np.argmax(raw[0], axis=0).astype(np.int32)
        else:
            class_map = raw[0].astype(np.int32)
        return [{"class_map": class_map}]

    def visualize(
        self,
        image_rgb: np.ndarray,
        results: list,
        alpha: float = 0.5,
        **kwargs,
    ) -> np.ndarray:
        class_map = results[0]["class_map"]
        h, w = image_rgb.shape[:2]

        # clip class_map to valid range in case model outputs unknown ids
        class_map = np.clip(class_map, 0, len(self.color_map) - 1)
        color_seg = self.color_map[class_map].astype(np.uint8)
        color_seg = cv2.resize(color_seg, (w, h), interpolation=cv2.INTER_NEAREST)

        return cv2.addWeighted(image_rgb, 1.0 - alpha, color_seg, alpha, 0)
