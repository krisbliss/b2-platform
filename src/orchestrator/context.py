"""Per-run session context passed to context-aware tools.

Threaded through pydantic-ai as the agent's ``deps`` so a tool can pull
out-of-band state — most importantly the user's most recent uploaded image —
without that data ever passing through the model. This keeps the mechanism
generic: any tool that declares ``needs_context`` receives this object and can
reach the transient store or the rendered conversation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SessionContext:
    """Runtime context for a single agent turn."""

    session_id: str | None = None
    store: Any | None = None          # FirestoreSessionStore (history + media)
    history_text: str = ""            # rendered prior conversation, used as narrative
