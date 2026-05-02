from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class ModelRunner(ABC):
    """
    Base class for Autoware ONNX model runners.

    Dict-based pipeline keeps preprocessing metadata (scale, padding, etc.)
    flowing through each stage without extra state:

        preprocess(input)        -> dict  (has 'tensor' + meta keys)
        infer(preprocess_result) -> dict  (adds 'outputs' key)
        postprocess(infer_result, **params) -> list[dict]
        visualize(original_input, results)  -> np.ndarray (RGB)

    load() accepts a unified config dict so each model can declare exactly
    which files it needs — ONNX, labels, color maps, anchor CSVs, YAML
    configs, etc. — without the base class prescribing a fixed schema.
    """

    def __init__(self) -> None:
        self.session = None
        self.config: dict = {}

    @abstractmethod
    def load(self, config: dict) -> None:
        """
        Load the ONNX model and all supporting files.

        Recognized config keys (convention, not enforced here):
          onnx        - path to .onnx file (required)
          label       - path to label.txt (optional)
          color_map   - path to id,name,r,g,b CSV (optional)
          param_files - dict of name -> path for additional parameter files
                        (anchor CSVs, prior-box npy, voxel config YAML, etc.)
          params      - dict of inline numeric / string parameters
                        (input_size, strides, mean, std, thresholds, …)
        """

    @abstractmethod
    def preprocess(self, input_data: Any) -> dict:
        """
        Convert raw input to model-ready tensors.
        Must return a dict containing at least 'tensor' plus any metadata
        needed downstream (e.g. scale, pad_left, pad_top, orig_w, orig_h).
        """

    @abstractmethod
    def infer(self, preprocess_result: dict) -> dict:
        """
        Run ONNX inference.
        Receives the dict from preprocess(), adds 'outputs' with raw model
        outputs, and returns the merged dict.
        """

    @abstractmethod
    def postprocess(self, infer_result: dict, **params) -> list:
        """
        Convert raw outputs to structured results.
        **params allows passing runtime values (conf_th, nms_th, alpha, …).
        Returns a list of result dicts whose schema is model-specific.
        """

    @abstractmethod
    def visualize(self, original_input: np.ndarray, results: list, **kwargs) -> np.ndarray:
        """
        Render results onto the original input.
        Returns an annotated RGB image (np.ndarray, dtype uint8).
        """
