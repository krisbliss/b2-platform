from typing import Optional


def handle(value: Optional[str] = None) -> str:
    """Return a simple example response for the example tool."""
    if value:
        return f"Example tool received: {value}"

    return "Example tool ran successfully."
