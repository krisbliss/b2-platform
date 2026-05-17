import asyncio

from tools.fake_image_detector.config_loader import CheckConfig, PipelineConfig
from tools.fake_image_detector.models import CheckResult, Escalation, Verdict
from tools.fake_image_detector.pipeline import FakeImageDetectorPipeline


def run(coro):
    return asyncio.run(coro)


class _StubCheck:
    def __init__(self, result: CheckResult):
        self._result = result
        self.calls = 0

    async def run(self, image_bytes: bytes, context: dict) -> CheckResult:
        self.calls += 1
        return self._result


def _cfg(check_id: str, early_exit_on_fail: bool = False) -> CheckConfig:
    return CheckConfig(id=check_id, enabled=True, early_exit_on_fail=early_exit_on_fail)


def _pipeline(
    checks: list[tuple[CheckConfig, _StubCheck]],
    gemini_check: _StubCheck | None = None,
) -> FakeImageDetectorPipeline:
    config = PipelineConfig(clear_fail=0.8, clear_pass=0.2, checks=[cfg for cfg, _ in checks])
    return FakeImageDetectorPipeline(config=config, checks=checks, gemini_check=gemini_check)


# helpers for common check shapes
def _passing(check_id: str) -> CheckResult:
    return CheckResult(check=check_id, passed=True, fake_score=0.0, confidence=0.8)


def _ambiguous(check_id: str) -> CheckResult:
    # score = 0.5*0.8 / 0.8 = 0.5 → ambiguous zone [0.2, 0.8)
    return CheckResult(check=check_id, passed=False, fake_score=0.5, confidence=0.8)


def _failing(check_id: str) -> CheckResult:
    # score = 1.0*0.9 / 0.9 = 1.0 → clear fail ≥ 0.8
    return CheckResult(check=check_id, passed=False, fake_score=1.0, confidence=0.9)


def test_unknown_input_type_flags_immediately():
    p = _pipeline([(_cfg("exif"), _StubCheck(_passing("exif")))])
    result = run(p.run(b"img", {"input_type": "unknown"}))
    assert result.verdict == Verdict.FLAG
    assert result.escalation == Escalation.HUMAN_REVIEW
    assert result.early_exit is True
    assert result.early_exit_reason == "UNKNOWN_INPUT_TYPE"
    assert result.checks == []


def test_document_route_runs_only_document_checks():
    exif = _StubCheck(_passing("exif"))
    ocr = _StubCheck(_passing("ocr_document"))
    ela = _StubCheck(_passing("ela"))
    p = _pipeline([(_cfg("exif"), exif), (_cfg("ocr_document"), ocr), (_cfg("ela"), ela)])
    result = run(p.run(b"img", {"input_type": "document"}))
    assert result.verdict == Verdict.PASS
    assert [r.check for r in result.checks] == ["exif", "ocr_document"]
    assert ela.calls == 0


def test_no_active_checks_for_route_flags():
    ela = _StubCheck(_passing("ela"))
    p = _pipeline([(_cfg("ela"), ela)])
    result = run(p.run(b"img", {"input_type": "document"}))
    assert result.verdict == Verdict.FLAG
    assert result.early_exit_reason == "NO_ACTIVE_CHECKS_FOR_DOCUMENT"
    assert ela.calls == 0


def test_check_runtime_error_flags_immediately():
    exif = _StubCheck(CheckResult(check="exif", passed=True, skipped=True, error="boom"))
    p = _pipeline([(_cfg("exif"), exif)])
    result = run(p.run(b"img", {"input_type": "face"}))
    assert result.verdict == Verdict.FLAG
    assert result.escalation == Escalation.HUMAN_REVIEW
    assert result.early_exit is True
    assert "CHECK_RUNTIME_ERROR" in result.checks[0].flags


def test_hard_escalation_flag_forces_human_review_before_score_classification():
    reverse_image = _StubCheck(
        CheckResult(
            check="reverse_image",
            passed=False,
            fake_score=0.95,
            confidence=0.95,
            flags=["FOUND_ONLINE"],
            human_escalate=True,
            escalation_reasons=["FOUND_ONLINE detected by reverse_image check"],
        )
    )
    p = _pipeline([(_cfg("reverse_image"), reverse_image)])
    result = run(p.run(b"img", {"input_type": "face"}))
    assert result.early_exit is True
    assert result.verdict == Verdict.FLAG
    assert result.escalation == Escalation.HUMAN_REVIEW
    assert "hard escalation" in (result.early_exit_reason or "")


def test_early_exit_on_fail_uses_fake_score():
    mrz = _StubCheck(CheckResult(check="mrz", passed=False, fake_score=1.0, confidence=1.0, flags=["MRZ_CHECKSUM_FAIL"]))
    p = _pipeline([(_cfg("mrz", early_exit_on_fail=True), mrz)])
    result = run(p.run(b"img", {"input_type": "document"}))
    assert result.early_exit is True
    assert result.verdict == Verdict.REJECT
    assert result.escalation == Escalation.AUTO_REJECT
    assert result.risk_score == 1.0


def test_gemini_called_in_ambiguous_zone():
    ela = _StubCheck(_ambiguous("ela"))
    gemini = _StubCheck(CheckResult(check="gemini_vision", passed=False, fake_score=0.6, confidence=0.85, flags=["GAN_ARTIFACTS"]))
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=gemini)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert gemini.calls == 1
    assert any(r.check == "gemini_vision" for r in result.checks)


def test_gemini_not_called_on_clear_pass():
    ela = _StubCheck(_passing("ela"))
    gemini = _StubCheck(CheckResult(check="gemini_vision", passed=False, fake_score=0.9, confidence=0.9))
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=gemini)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert gemini.calls == 0
    assert result.verdict == Verdict.PASS


def test_gemini_not_called_on_clear_fail():
    ela = _StubCheck(_failing("ela"))
    gemini = _StubCheck(CheckResult(check="gemini_vision", passed=False, fake_score=0.95, confidence=0.95))
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=gemini)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert gemini.calls == 0
    assert result.verdict == Verdict.REJECT


def test_gemini_skipped_returns_stage1_with_skip_record():
    ela = _StubCheck(_ambiguous("ela"))
    gemini = _StubCheck(CheckResult(check="gemini_vision", passed=True, skipped=True))
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=gemini)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert result.verdict == Verdict.FLAG
    assert result.risk_score == 0.5
    assert any(r.check == "gemini_vision" and r.skipped for r in result.checks)


def test_gemini_hard_escalation_overrides_clear_reject():
    ela = _StubCheck(_ambiguous("ela"))
    gemini = _StubCheck(
        CheckResult(
            check="gemini_vision",
            passed=False,
            fake_score=0.95,
            confidence=0.95,
            flags=["POSSIBLE_STOCK"],
            human_escalate=True,
            escalation_reasons=["POSSIBLE_STOCK detected by gemini_vision check"],
        )
    )
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=gemini)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert result.verdict == Verdict.FLAG
    assert result.escalation == Escalation.HUMAN_REVIEW
    assert result.early_exit is True


def test_gemini_real_verdict_overrides_ambiguous_stage1():
    ela = _StubCheck(_ambiguous("ela"))
    gemini = _StubCheck(CheckResult(check="gemini_vision", passed=True, fake_score=0.05, confidence=0.9))
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=gemini)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert result.verdict == Verdict.PASS
    assert result.risk_score == 0.05


def test_gemini_none_returns_stage1():
    ela = _StubCheck(_ambiguous("ela"))
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=None)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert result.verdict == Verdict.FLAG
    assert result.risk_score == 0.5


def test_gemini_always_called_for_documents():
    # Documents always go to Gemini regardless of stage-1 score.
    ocr = _StubCheck(_passing("ocr_document"))
    gemini = _StubCheck(CheckResult(check="gemini_vision", passed=True, fake_score=0.1, confidence=0.9))
    p = _pipeline([(_cfg("ocr_document"), ocr)], gemini_check=gemini)
    result = run(p.run(b"img", {"input_type": "document"}))
    assert gemini.calls == 1
    assert any(r.check == "gemini_vision" for r in result.checks)
    assert result.verdict == Verdict.PASS
    assert result.risk_score == 0.1


def test_gemini_not_called_for_face_on_clear_fail():
    # For faces, a clear REJECT bypasses Gemini (no need to spend API calls).
    ela = _StubCheck(_failing("ela"))
    gemini = _StubCheck(CheckResult(check="gemini_vision", passed=False, fake_score=0.95, confidence=0.95))
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=gemini)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert gemini.calls == 0
    assert result.verdict == Verdict.REJECT


def test_zero_confidence_stage1_results_flag_for_review():
    ela = _StubCheck(CheckResult(check="ela", passed=True, fake_score=0.0, confidence=0.0))
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=None)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert result.risk_score == 0.21
    assert result.verdict == Verdict.FLAG
    assert result.escalation == Escalation.HUMAN_REVIEW
