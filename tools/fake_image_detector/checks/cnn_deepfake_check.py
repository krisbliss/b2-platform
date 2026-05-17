import asyncio
import io
from pathlib import Path

import numpy as np

from tools.fake_image_detector.checks.base_check import BaseCheck
from tools.fake_image_detector.models import CheckResult, NormalizedSignals

_MODEL_DIR = Path(__file__).parent.parent / "models"
_DEFAULT_MODEL = _MODEL_DIR / "cnn_deepfake.onnx"
_FALLBACK_MODEL = _MODEL_DIR / "cnn_deepfake_int8.onnx"

# ViT preprocessing constants (preprocessor_config.json from HuggingFace)
_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_INPUT_SIZE = 224

# P(deepfake) above this threshold → flag
_THRESHOLD = 0.5


def _find_model() -> Path | None:
    for candidate in (_DEFAULT_MODEL, _FALLBACK_MODEL):
        if candidate.exists():
            return candidate
    return None


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class CNNDeepfakeCheck(BaseCheck):
    check_id = "cnn_deepfake"

    def __init__(self, params: dict | None = None, model_path: Path | None = None):
        self._model_path = model_path or _find_model()
        self._session = None

    def _get_session(self):
        if self._session is not None:
            return self._session
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # suppress verbose ONNX logs
        self._session = ort.InferenceSession(str(self._model_path), sess_options=opts)
        return self._session

    async def run(self, image_bytes: bytes, context: dict) -> CheckResult:
        return await asyncio.to_thread(self._run_sync, image_bytes)

    def _run_sync(self, image_bytes: bytes) -> CheckResult:
        if self._model_path is None or not self._model_path.exists():
            return CheckResult(
                check=self.check_id,
                passed=True,
                confidence=0.0,
                skipped=True,
                error="MODEL_NOT_AVAILABLE",
            )

        try:
            from PIL import Image
        except ImportError as e:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True, error=str(e))

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize((_INPUT_SIZE, _INPUT_SIZE), Image.Resampling.BILINEAR)
        except Exception as e:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True, error=str(e))

        try:
            session = self._get_session()
        except Exception as e:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True, error=str(e))

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        tensor = arr.transpose(2, 0, 1)[np.newaxis]  # HWC → NCHW

        try:
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            logits = session.run([output_name], {input_name: tensor})[0][0]
        except Exception as e:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True, error=str(e))

        probs = _softmax(logits.astype(np.float32))
        p_fake = float(probs[0])  # index 0 = Deepfake
        p_real = float(probs[1])  # index 1 = Real

        decisiveness = round(max(p_fake, p_real), 3)  # how far from 50/50

        if p_fake >= _THRESHOLD:
            return CheckResult(
                check=self.check_id,
                passed=False,
                fake_score=round(p_fake, 3),
                confidence=decisiveness,
                flags=["CNN_DEEPFAKE_DETECTED"],
                signals={"p_fake": round(p_fake, 3), "p_real": round(p_real, 3)},
                normalized_signals=NormalizedSignals(
                    category="synthetic",
                    confidence=decisiveness,
                    indicators=["CNN_DEEPFAKE_DETECTED"],
                    synthetic_score=round(p_fake, 3),
                ),
            )

        return CheckResult(
            check=self.check_id,
            passed=True,
            fake_score=round(p_fake, 3),
            confidence=decisiveness,
            signals={"p_fake": round(p_fake, 3), "p_real": round(p_real, 3)},
            normalized_signals=NormalizedSignals(
                category="synthetic",
                confidence=decisiveness,
                indicators=["CLEAN"],
                synthetic_score=round(p_fake, 3),
            ),
        )
