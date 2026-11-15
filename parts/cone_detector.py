"""Cone detector: YOLO26n-p2 trained to find traffic cones as obstacles.

How the model was trained and evaluated is in cone_detection/README.md.

    from parts.cone_detector import Cone_Detector
    V.add(Cone_Detector(), inputs=['sensors/ZED/RGB/left'], outputs=['perception/cones'])

run(image) returns a float32 array of shape (N, 5), one row per cone:
x1, y1, x2, y2, confidence, in the INPUT image's pixels, most confident first.
It returns an empty (0, 5) array when there is no image or no cone.

These are pixel boxes only. Turning them into world positions with the ZED
depth map is the remaining step of issue #4.

On the Jetson, build the TensorRT engine once with
cone_detection/scripts/jetson/export_engine.sh; this part then picks it up
automatically (about 29 ms per frame on the Orin Nano Super). Without the
engine it falls back to the PyTorch .pt, which works anywhere but is slower.
"""
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "cone_detection" / "models"
EMPTY = np.zeros((0, 5), np.float32)


class Cone_Detector:
    def __init__(self, model=None, imgsz=(736, 1280), conf=0.25, device=None):
        """
        Args:
            model: path to a .engine or .pt. Default: the engine next to the
                trained .pt if it has been built, else the .pt.
            imgsz: (height, width) the model sees. Must match the engine;
                (736, 1280) fits the ZED's 1280x720 HD720 frames.
            conf: confidence threshold. Kept low on purpose - a missed cone is
                a collision, while a false alarm can be filtered over frames.
            device: None picks the GPU when there is one.
        """
        # imported here so runners that don't use this part don't need ultralytics
        from ultralytics import YOLO

        if model is None:
            engine = MODELS / "base_p2_1280.engine"
            model = engine if engine.exists() else MODELS / "base_p2_1280.pt"
        self.model = YOLO(str(model), task="detect")
        self.imgsz, self.conf, self.device = imgsz, conf, device
        print(f"cone detector loaded: {model}")

    def run(self, image):
        if image is None:
            return EMPTY
        if image.ndim == 3 and image.shape[2] == 4:
            # the ZED returns BGRA; the model expects BGR
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf,
                               device=self.device, verbose=False)[0]
        if len(r.boxes) == 0:
            return EMPTY
        boxes = np.concatenate([r.boxes.xyxy.cpu().numpy(),
                                r.boxes.conf.cpu().numpy()[:, None]], axis=1)
        return boxes[np.argsort(-boxes[:, 4])].astype(np.float32)

    def shutdown(self):
        pass
