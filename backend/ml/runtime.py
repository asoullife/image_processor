import importlib
import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)
_probe_cache = None


def probe():
    """Detect available ML libraries and hardware.

    Returns a SimpleNamespace with attributes:
    - tf: TensorFlow module or None
    - yolo: ultralytics module or None
    - cv2: OpenCV module or None
    - torch: PyTorch module or None
    - cuda: bool indicating if CUDA is available
    - device: "cuda" or "cpu" depending on CUDA availability
    """
    global _probe_cache
    if _probe_cache is not None:
        return _probe_cache

    def _try_import(name: str):
        try:
            module = importlib.import_module(name)
            logger.info("%s available", name)
            return module
        except Exception as e:
            logger.info("%s not available: %s", name, e)
            return None

    tf = _try_import("tensorflow")
    ultralytics = _try_import("ultralytics")
    cv2 = _try_import("cv2")
    torch = _try_import("torch")

    cuda = False
    if torch is not None:
        try:
            cuda = torch.cuda.is_available()
        except Exception as e:
            logger.info("torch cuda check failed: %s", e)

    device = "cuda" if cuda else "cpu"

    _probe_cache = SimpleNamespace(
        tf=tf,
        yolo=ultralytics,
        cv2=cv2,
        torch=torch,
        cuda=cuda,
        device=device,
    )

    logger.info(
        "ML runtime probe: TF=%s, YOLO=%s, OpenCV=%s, CUDA=%s",
        bool(tf),
        bool(ultralytics),
        bool(cv2),
        cuda,
    )
    return _probe_cache


def get_detector(model: str = "yolov8n"):
    """Load a YOLOv8 detector on the optimal device."""
    info = probe()
    if not info.yolo:
        raise RuntimeError("ultralytics YOLOv8 not installed")

    YOLO = getattr(info.yolo, "YOLO", None)
    if YOLO is None:
        raise RuntimeError("YOLO class not found in ultralytics package")

    try:
        detector = YOLO(model)
        if info.cuda and hasattr(detector, "to"):
            detector.to("cuda")
        return detector
    except Exception as e:
        logger.error("Failed to load YOLO model %s: %s", model, e)
        raise
