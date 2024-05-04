from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional
import numpy as np
from utils.device_util import select_device
from utils.vision_utils import (
    reshape_image,
    non_max_suppression,
    scale_boxes,
    Detection,
)
import torch
from utils.logger import Logger, LogLevel
from os import environ
import time
import cv2
import colorsys
from PIL import Image

# region Attempt ML framework imports
FLAG_ONNX_AVAILABLE = False
FLAG_TORCH_AVAILABLE = False
FLAG_TENSORRT_AVAILABLE = False
try:
    import onnxruntime as ort

    FLAG_ONNX_AVAILABLE = True
except ImportError:
    pass
try:
    import torch

    FLAG_TORCH_AVAILABLE = True
except ImportError:
    pass
try:
    import tensorrt as trt

    FLAG_TENSORRT_AVAILABLE = True
except ImportError:
    pass
# endregion


class BaseDetector(ABC):

    def __init__(
        self,
        model_path: str,
        classes: List[str],
        image_size: Tuple[int, int] = (640, 640),
        *args,
        **kwargs,
    ) -> None:
        log_level = LogLevel.from_str(
            kwargs.get("log_level", environ.get("LOG_LEVEL", "INFO"))
        )
        self.logger = Logger(
            basename=kwargs.get("basename", "BaseDetector"),
            log_level=log_level,
            console=kwargs.get("console", None),
        )
        self.classes = classes
        self.image_size = image_size
        self.device = select_device(kwargs.get("device", 0))
        self.logger.info("selected device for inference:" + str(self.device))
        self.conf_thres = kwargs.get("conf_thres", 0.25)
        self.half = kwargs.get("half", None)
        self.iou_thres = kwargs.get("iou_thres", 0.45)
        self.agnostic_nms = kwargs.get("agnostic_nms", False)
        self.max_detections = kwargs.get("max_detections", 50)
        self.warn = kwargs.get("warn", True)
        self.warmup = kwargs.get("warmup", True)
        self.half = kwargs.get("half", None)
        if self.device.type != "cpu" and self.half:
            self.half = False
        if self.half:
            self.logger.info("half precision enabled")
        self.load_model(model_path)
        if self.warmup:
            self._warmup()
        n_classes = len(classes)
        hsv_tuples = [(1.0 * x / n_classes, 1.0, 1.0) for x in range(n_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(
                lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors,
            )
        )

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> torch.Tensor:
        pass

    def _warmup(self, *args, **kwargs) -> None:
        self.logger.info("warming up model ...")
        sample_input_zeros = None
        sample_input_zeros = np.zeros(
            (1, 3, *self.image_size), dtype=np.float16 if self.half else np.float32
        )
        for _ in range(3):
            _ = self.forward(sample_input_zeros)

    def draw_and_collect_bbox(
        self, image: np.ndarray, bboxes: np.ndarray
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
        """
        bbox_collection = []
        image_h, image_w, _ = image.shape
        fontScale = 0.5
        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_ind = int(bbox[5])
            bbox_color = self.colors[class_ind]
            bbox_thick = int(0.85 * (image_h + image_w) / 600)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
            label = self.classes[class_ind]
            bbox_mess = "%s: %.2f" % (label, score)
            t_size = cv2.getTextSize(
                bbox_mess, 0, fontScale, thickness=bbox_thick // 2
            )[0]
            cv2.rectangle(
                image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1
            )
            cv2.putText(
                image,
                bbox_mess,
                (c1[0], c1[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                (0, 0, 0),
                bbox_thick // 2,
                lineType=cv2.LINE_AA,
            )
            bbox_collection.append(
                Detection(
                    xy1=(coor[0], coor[1]),
                    xy2=(coor[2], coor[3]),
                    label=label,
                    prob=score,
                )
            )
        return image, bbox_collection

    def predict(
        self,
        source: Union[str, np.ndarray],
        isBGR: bool = False,
        *args,
        **kwargs,
    ) -> None:
        original_image = None
        scaled_inference = False
        img0 = cv2.imread(source) if isinstance(source, str) else source
        if isBGR:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        if img0.shape[:2] != self.image_size:
            scaled_inference = True
            original_image = np.copy(img0)
            if self.warn:
                self.logger.warning(
                    "Image is being shaped, check gstreamer pipeline settings"
                )
            img0, _, _ = reshape_image(img0, new_shape=self.image_size, stride=32)
        img = np.copy(img0)
        img = img.transpose((2, 0, 1)) / 255  # HWC to CHW
        # img is of type float64 & has shape (3, image_size[0], image_size[1]), so need to convert to float32 & add a batch dimension
        img = np.expand_dims(img, axis=0).astype("float32")
        preds = self.forward(img)
        self.logger.debug(f"Predictions Shape: {preds.shape}")
        nms_preds = non_max_suppression(
            preds,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            agnostic=self.agnostic_nms,
            max_det=self.max_detections,
        )[0]
        if scaled_inference:
            nms_preds[:, :4] = scale_boxes(img.shape[2:], nms_preds[:, :4], img0.shape)
            return self.draw_and_collect_bbox(original_image, nms_preds)
        return self.draw_and_collect_bbox(img0, nms_preds)

    @staticmethod
    def save_np_image(image, output_path):
        image = Image.fromarray(image)
        image.save(output_path)


class ONNXDetector(BaseDetector):

    DEFAULT_ONNX_PROVIDERS = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    def __init__(
        self,
        model_path: str,
        classes: List[str],
        image_size: Tuple[int, int] = (640, 640),
        onnx_providers: List[str] = DEFAULT_ONNX_PROVIDERS,
        *args,
        **kwargs,
    ) -> None:
        if not FLAG_ONNX_AVAILABLE:
            raise ImportError("ONNXRuntime not installed.")
        kwargs.update(
            {
                "basename": "ONNXDetector",
            }
        )
        self.onnx_providers = onnx_providers
        super(ONNXDetector, self).__init__(
            model_path, classes, image_size, *args, **kwargs
        )
        self.model = None

    def load_model(self, model_path: str) -> None:
        start = time.time()
        self.logger.info(f"loading model from {model_path}")
        self.sess = ort.InferenceSession(model_path, providers=self.onnx_providers)
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name
        self.logger.info(f"model loaded in {time.time() - start:.2f}s")

    def forward(self, x: np.ndarray) -> torch.Tensor:
        if self.half:
            x = x.astype(np.float16)
        y = self.sess.run([self.output_name], {self.input_name: x})
        y = torch.tensor(y[0])
        return y


if __name__ == "__main__":

    environ["LOG_LEVEL"] = "DEBUG"
    model_path = r"models/best.onnx"
    classes = ["face", "human"]
    sample_image = r"data/samples/9.jpg"
    output_file = r"data/outputs/9_op.jpg"
    # Create an instance of ONNXDetector
    detector = ONNXDetector(model_path=model_path, classes=classes, device="cpu")
    img1, detections = detector.predict(source=sample_image, isBGR=True)
    detector.save_np_image(img1, output_file)
    detector.save_np_image(img1, output_file)
    print("done")
