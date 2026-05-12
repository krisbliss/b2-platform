import logging

from .router import AgentRouter, BelowThresholdError
from .session import Session

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def chat_loop() -> None:
	"""Run a simple terminal chat loop against the routed agent."""
	router = AgentRouter()
	active_session: Session | None = None

	while True:
		try:
			query = input("You: ").strip()
		except KeyboardInterrupt:
			print("\nInterrupted. Type '/route' to reroute or 'exit' to quit.")
			continue
		except EOFError:
			print("\nGoodbye.")
			break

		if not query:
			continue

		if query.lower() in {"exit", "quit"}:
			print("Goodbye.")
			break

		if query == "/route":
			active_session = None
			print("Routing reset. Your next message will select an agent.")
			continue

		if active_session is None:
			try:
				agent, metadata = router.route_with_metadata(query)
			except BelowThresholdError:
				print("Router: I'm not sure which agent fits - try rephrasing, or type /route to reset.")
				continue
			except Exception as exc:
				print(f"Router: {exc}")
				continue
			active_session = Session(agent)
			print(f"Routed to agent: {agent.name} (score={metadata['score']:.3f})")

		print("Agent: ", end="", flush=True)
		for chunk in active_session.send_stream(query):
			print(chunk, end="", flush=True)
		print()


def main() -> None:
	"""Entry point for the interactive loop."""
	chat_loop()
