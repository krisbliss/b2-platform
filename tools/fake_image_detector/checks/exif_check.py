import asyncio
import re

import piexif

from tools.fake_image_detector.checks.base_check import BaseCheck, CheckContext
from tools.fake_image_detector.models import CheckResult

EDITING_SOFTWARE_PATTERN = re.compile(r"\b(?:photoshop|gimp|lightroom|affinity|capture one|darktable)\b")


class EXIFCheck(BaseCheck):
    check_id = "exif"

    def __init__(self, params: dict | None = None) -> None:
        p = params or {}
        self._no_exif_confidence = float(p.get("no_exif_confidence", 0.2))

    async def run(self, image_bytes: bytes, context: CheckContext) -> CheckResult:
        return await asyncio.to_thread(self._run_sync, image_bytes)

    def _run_sync(self, image_bytes: bytes) -> CheckResult:
        try:
            exif_data = piexif.load(image_bytes)
        except (piexif.InvalidImageDataError, ValueError):
            # ValueError covers PNGs and other non-JPEG formats piexif can't parse
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True)

        zeroth = exif_data.get("0th", {})
        software_raw = zeroth.get(piexif.ImageIFD.Software)

        if not zeroth and not exif_data.get("Exif"):
            return CheckResult(
                check=self.check_id,
                passed=True,
                fake_score=0.0,
                confidence=self._no_exif_confidence,
                flags=["NO_EXIF_DATA"],
            )

        if software_raw:
            software = software_raw.decode("utf-8", errors="ignore").strip().lower()
            if EDITING_SOFTWARE_PATTERN.search(software):
                return CheckResult(
                    check=self.check_id,
                    passed=False,
                    fake_score=1.0,
                    confidence=1.0,
                    flags=["EDITING_SOFTWARE_DETECTED"],
                    signals={"software": software},
                )

        make = zeroth.get(piexif.ImageIFD.Make, b"").decode("utf-8", errors="ignore").strip()
        model = zeroth.get(piexif.ImageIFD.Model, b"").decode("utf-8", errors="ignore").strip()

        return CheckResult(
            check=self.check_id,
            passed=True,
            fake_score=0.0,
            confidence=1.0,
            signals={"make": make or None, "model": model or None},
        )
