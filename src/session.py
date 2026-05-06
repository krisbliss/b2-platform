import logging
from time import perf_counter

from pydantic_ai.messages import ModelMessage

from .orchestrator.agent import Agent

logger = logging.getLogger(__name__)


class Session:
    def __init__(self, agent: Agent):
        self.agent = agent
        self._history: list[ModelMessage] = []

    def send(self, user_message: str) -> str:
        start = perf_counter()
        logger.info("session.send start history=%d chars=%d", len(self._history), len(user_message))
        result = self.agent.run(user_message, message_history=self._history)
        logger.info("session.send model returned elapsed=%.3fs", perf_counter() - start)
        self._history = result.all_messages()
        logger.info(
            "session.send done elapsed=%.3fs history=%d",
            perf_counter() - start,
            len(self._history),
        )
        return result.output

    def send_stream(self, user_message: str):
        start = perf_counter()
        logger.info("session.stream start history=%d chars=%d", len(self._history), len(user_message))
        streamed = self.agent.run_stream(user_message, message_history=self._history)
        logger.info("session.stream object returned elapsed=%.3fs", perf_counter() - start)
        saw_delta = False
        first_chunk_at: float | None = None

        for chunk in streamed.stream_text(delta=True, debounce_by=None):
            if first_chunk_at is None:
                first_chunk_at = perf_counter()
                logger.info("session.stream first chunk elapsed=%.3fs", first_chunk_at - start)
            saw_delta = True
            yield chunk

        self._history = streamed.all_messages()

        if not saw_delta:
            output = streamed.get_output()
            if output:
                logger.info("session.stream fallback output elapsed=%.3fs", perf_counter() - start)
                yield str(output)
            else:
                tool_results = self._latest_tool_results(self._history)
                if tool_results:
                    logger.info(
                        "session.stream fallback tool results elapsed=%.3fs count=%d",
                        perf_counter() - start,
                        len(tool_results),
                    )
                    yield "\n".join(tool_results)

        logger.info(
            "session.stream done elapsed=%.3fs chunks=%s history=%d",
            perf_counter() - start,
            "yes" if saw_delta else "no",
            len(self._history),
        )

    @staticmethod
    def _latest_tool_results(messages: list[ModelMessage]) -> list[str]:
        results: list[str] = []
        for message in reversed(messages):
            parts = getattr(message, "parts", [])
            for part in reversed(parts):
                if type(part).__name__ != "ToolReturnPart":
                    continue
                tool_name = getattr(part, "tool_name", "tool")
                content = getattr(part, "content", "")
                if isinstance(content, str):
                    text = content
                else:
                    text = str(content)
                results.append(f"{tool_name}: {text}")
            if results:
                return list(reversed(results))
        return results
