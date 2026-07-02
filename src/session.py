import logging
from time import perf_counter
from typing import Sequence

from pydantic_ai.messages import ModelMessage, UserContent

from .orchestrator.agent import Agent

logger = logging.getLogger(__name__)


class Session:
    def __init__(self, agent: Agent, history: Sequence[ModelMessage] | None = None):
        self.agent = agent
        self._history: list[ModelMessage] = list(history or [])

    @property
    def history(self) -> list[ModelMessage]:
        return list(self._history)

    def send_stream(self, user_message: str | Sequence[UserContent]):
        start = perf_counter()
        logger.info("session.stream start history=%d input_type=%s", len(self._history), type(user_message).__name__)
        streamed = self.agent.run_stream(user_message, message_history=self._history)
        logger.info("session.stream object returned elapsed=%.3fs", perf_counter() - start)
        first_chunk_at: float | None = None

        for chunk in streamed.stream_text(delta=True, debounce_by=None):
            if first_chunk_at is None:
                first_chunk_at = perf_counter()
                logger.info("session.stream first chunk elapsed=%.3fs", first_chunk_at - start)
            yield chunk

        self._history = streamed.all_messages()

        logger.info(
            "session.stream done elapsed=%.3fs chunks=%s history=%d",
            perf_counter() - start,
            "yes" if first_chunk_at is not None else "no",
            len(self._history),
        )
