import asyncio
import io

import numpy as np
from PIL import Image, ImageChops, UnidentifiedImageError

from tools.fake_image_detector.checks.base_check import BaseCheck, CheckContext
from tools.fake_image_detector.models import CheckResult, NormalizedSignals

_DEFAULT_THRESHOLD = 4.0
_RESAVE_QUALITY = 95


class ELACheck(BaseCheck):
    check_id = "ela"

    def __init__(self, params: dict | None = None) -> None:
        p = params or {}
        self._threshold = float(p.get("threshold", _DEFAULT_THRESHOLD))
        self._document_threshold = float(p.get("document_threshold", self._threshold))

    async def run(self, image_bytes: bytes, context: CheckContext) -> CheckResult:
        return await asyncio.to_thread(self._run_sync, image_bytes, context)

    def _run_sync(self, image_bytes: bytes, context: CheckContext) -> CheckResult:
        threshold = self._document_threshold if context.get("input_type") == "document" else self._threshold

        try:
            source = Image.open(io.BytesIO(image_bytes))
            if "A" in source.getbands():
                alpha = source.convert("RGBA")
                original = Image.new("RGB", alpha.size, (255, 255, 255))
                original.paste(alpha, mask=alpha.getchannel("A"))
            else:
                original = source.convert("RGB")
        except (UnidentifiedImageError, ValueError, OSError) as e:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True, error=str(e))

        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=_RESAVE_QUALITY)
        buf.seek(0)
        resaved = Image.open(buf).convert("RGB")

        diff = ImageChops.difference(original, resaved)
        amplified = diff.point(lambda px: min(px * 20, 255))

        ela_mean = float(np.array(amplified).mean())
        ela_max = int(np.array(amplified).max())

        if ela_mean > threshold:
            return CheckResult(
                check=self.check_id,
                passed=False,
                fake_score=1.0,
                confidence=1.0,
                flags=["HIGH_ELA_SCORE"],
                signals={"ela_mean": round(ela_mean, 3), "ela_max": ela_max},
                normalized_signals=NormalizedSignals(
                    category="manipulation",
                    confidence=1.0,
                    indicators=["HIGH_ELA_SCORE"],
                    manipulation_score=1.0,
                ),
            )

        return CheckResult(
            check=self.check_id,
            passed=True,
            fake_score=0.0,
            confidence=1.0,
            signals={"ela_mean": round(ela_mean, 3), "ela_max": ela_max},
            normalized_signals=NormalizedSignals(
                category="manipulation",
                confidence=1.0,
                indicators=["CLEAN"],
                manipulation_score=0.0,
            ),
        )
