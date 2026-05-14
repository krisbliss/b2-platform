from abc import ABC, abstractmethod
from typing import Literal, TypedDict

from tools.fake_image_detector.models import CheckResult


class CheckContext(TypedDict, total=False):
    input_type: Literal["document", "face", "unknown"]
    doc_type: str
    country: str
    extracted_fields: dict[str, str]


class BaseCheck(ABC):
    check_id: str = ""

    def __init__(self, params: dict | None = None) -> None:
        pass

    @abstractmethod
    async def run(self, image_bytes: bytes, context: CheckContext) -> CheckResult:
        ...
