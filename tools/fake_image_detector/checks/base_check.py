from abc import ABC, abstractmethod
from typing import TypedDict

from tools.fake_image_detector.models import CheckResult


class CheckContext(TypedDict, total=False):
    pass


class BaseCheck(ABC):
    check_id: str = ""

    def __init__(self, params: dict | None = None) -> None:
        pass

    @abstractmethod
    async def run(self, image_bytes: bytes, context: CheckContext) -> CheckResult:
        ...
