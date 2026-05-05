from typing import Optional

from router import route_agent


def chat_loop() -> None:
	"""Run a simple terminal chat loop against the routed agent."""
	active_agent = None

	while True:
		query = input("You: ").strip()

		if not query:
			continue

		if query.lower() in {"exit", "quit"}:
			print("Goodbye.")
			break

		if active_agent is None:
			active_agent = route_agent(query)
			print(f"Routed to agent: {active_agent.name}")

		response = active_agent.call(query)
		print(f"Agent: {response}")


def main() -> None:
	"""Entry point for the interactive loop."""
	chat_loop()
