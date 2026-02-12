"""Product detector for shelf images using YOLOWorld (open-vocabulary)."""

import dataclasses
from PIL import Image

from src.config import DETECTOR_CONF_THRESHOLD, DETECTOR_MIN_BOX_AREA, DETECTOR_BOX_PADDING, DETECTOR_CLASSES


@dataclasses.dataclass
class Detection:
    """A single detected object in a shelf image."""
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords
    score: float                     # detector confidence
    class_id: int                    # COCO class index
    class_name: str                  # COCO class name
    crop: Image.Image               # cropped PIL image

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> int:
        return self.width * self.height


class Detector:
    """Detect product candidates in images using YOLOWorld (open-vocabulary)."""

    def __init__(self):
        self._model = None

    def load(self) -> None:
        """Load YOLOWorld with open-vocabulary grocery classes."""
        from ultralytics import YOLOWorld
        self._model = YOLOWorld("yolov8s-worldv2.pt")
        self._model.set_classes(DETECTOR_CLASSES)

    def detect(
        self,
        image: Image.Image,
        conf_threshold: float = DETECTOR_CONF_THRESHOLD,
        min_area: int = DETECTOR_MIN_BOX_AREA,
    ) -> list[Detection]:
        """Run detection on a PIL image.

        Args:
            image: Full shelf image (PIL RGB).
            conf_threshold: Minimum confidence to keep a detection.
            min_area: Minimum box area in pixels (filters tiny false positives).

        Returns:
            List of Detections sorted left-to-right by x-coordinate.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        image = image.convert("RGB")
        img_w, img_h = image.size

        results = self._model(image, conf=conf_threshold, verbose=False)
        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                score = float(boxes.conf[i])
                class_id = int(boxes.cls[i])
                class_name = self._model.names[class_id]

                # Convert to int pixel coords
                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x2), int(y2)

                # Filter small boxes
                if (x2 - x1) * (y2 - y1) < min_area:
                    continue

                # Crop with padding
                pad = DETECTOR_BOX_PADDING
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(img_w, x2 + pad)
                cy2 = min(img_h, y2 + pad)
                crop = image.crop((cx1, cy1, cx2, cy2))

                detections.append(Detection(
                    box=(x1, y1, x2, y2),
                    score=score,
                    class_id=class_id,
                    class_name=class_name,
                    crop=crop,
                ))

        # Sort left-to-right
        detections.sort(key=lambda d: d.box[0])
        return detections
