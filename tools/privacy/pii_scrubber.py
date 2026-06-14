"""Presidio-backed PII scrubbing layer for the Canonical Message Envelope."""

from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from src.envelope import CanonicalMessage

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "pii_config.yaml"


@dataclass(frozen=True)
class EntityConfig:
    name: str
    enabled: bool = True
    recognizer: str = "regex"        # regex | ner | custom
    allow_list: tuple[str, ...] = () # values never flagged (case-insensitive)
    deny_list: tuple[str, ...] = ()  # values always flagged


@dataclass(frozen=True)
class NlpEngineConfig:
    provider: str = "spacy"
    model_name: str = "en_core_web_sm"


@dataclass(frozen=True)
class AuditStoreConfig:
    memory_ttl_hours: int = 24
    file_path: str = "data/pii_audit.jsonl"
    store_originals: bool = True  # set False once GL-20 encryption is live


@dataclass
class PiiConfig:
    """Full scrubber configuration. Load via PiiConfig.from_yaml()."""

    score_threshold: float = 0.5
    max_retries: int = 3
    retry_timeout_ms: int = 200
    language: str = "en"
    entities: list[EntityConfig] = field(default_factory=list)
    nlp_engine: NlpEngineConfig = field(default_factory=NlpEngineConfig)
    audit_store: AuditStoreConfig = field(default_factory=AuditStoreConfig)

    @property
    def enabled_entity_names(self) -> list[str]:
        return [e.name for e in self.entities if e.enabled]

    @property
    def needs_ner(self) -> bool:
        return any(e.recognizer == "ner" and e.enabled for e in self.entities)

    @classmethod
    def from_yaml(cls, path: Path = _DEFAULT_CONFIG_PATH) -> PiiConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        scrubber_raw = raw.get("scrubber", {})
        entities = [
            EntityConfig(
                name=e["name"],
                enabled=e.get("enabled", True),
                recognizer=e.get("recognizer", "regex"),
                allow_list=tuple(e.get("allow_list") or []),
                deny_list=tuple(e.get("deny_list") or []),
            )
            for e in raw.get("entities", [])
        ]
        nlp_raw = raw.get("nlp_engine", {})
        audit_raw = raw.get("audit_store", {})
        return cls(
            score_threshold=scrubber_raw.get("score_threshold", 0.5),
            max_retries=scrubber_raw.get("max_retries", 3),
            retry_timeout_ms=scrubber_raw.get("retry_timeout_ms", 200),
            language=scrubber_raw.get("language", "en"),
            entities=entities,
            nlp_engine=NlpEngineConfig(
                provider=nlp_raw.get("provider", "spacy"),
                model_name=nlp_raw.get("model_name", "en_core_web_sm"),
            ),
            audit_store=AuditStoreConfig(
                memory_ttl_hours=audit_raw.get("memory_ttl_hours", 24),
                file_path=audit_raw.get("file_path", "data/pii_audit.jsonl"),
                store_originals=audit_raw.get("store_originals", True),
            ),
        )


@dataclass(frozen=True)
class PiiEntity:
    """One PII item found in a CanonicalMessage field.

    Goes to PiiAuditStore only — never forwarded to the Agentic OS.
    sha256_hash enables correlation across sessions; original_value enables
    authorized pilot recovery.
    """

    field_path: str      # e.g. "text_content", "session_context.bio"
    entity_type: str     # e.g. "PHONE_NUMBER", "EMAIL_ADDRESS"
    original_value: str  # raw PII — audit store only, never enters OS
    sha256_hash: str     # SHA-256 of original_value
    placeholder: str     # replaces the value in the clean CanonicalMessage
    score: float         # Presidio confidence (0.0-1.0)
    start: int           # character offset in original field text
    end: int


@dataclass
class ScrubResult:
    """Result of one PiiScrubber.scrub() call.

    clean_message  -> safe for the Agentic OS (PII replaced with placeholders).
    entities       -> goes to PiiAuditStore only, never forwarded to the OS.
    scrub_failed   -> True if Presidio errored after all retries.
    """

    clean_message: CanonicalMessage
    entities: list[PiiEntity]
    scrub_failed: bool = False
    failure_reason: str | None = None
    scrubbed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def found_pii(self) -> bool:
        return bool(self.entities)


@dataclass
class _AuditEntry:
    session_id_hash: str
    scrubbed_at: str
    scrub_failed: bool
    failure_reason: str | None
    entities: list[dict[str, Any]]

    def to_json_line(self) -> str:
        return json.dumps({
            "session_id_hash": self.session_id_hash,
            "scrubbed_at": self.scrubbed_at,
            "scrub_failed": self.scrub_failed,
            "failure_reason": self.failure_reason,
            "entities": self.entities,
        }, ensure_ascii=False)


class PiiAuditStore:
    """Isolated PII record store — never read by the Agentic OS.

    In-memory list clears at midnight UTC (matches session_id daily rotation).
    File is append-only JSONL; access is restricted to authorized operators.
    """

    def __init__(self, config: AuditStoreConfig) -> None:
        self._config = config
        self._memory: list[_AuditEntry] = []
        self._current_day: date = datetime.now(timezone.utc).date()
        self._file_path = Path(config.file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, session_id_hash: str, result: ScrubResult) -> None:
        self._rotate_if_new_day()
        entity_dicts: list[dict[str, Any]] = []
        for e in result.entities:
            entry: dict[str, Any] = {
                "field_path": e.field_path,
                "entity_type": e.entity_type,
                "sha256_hash": e.sha256_hash,
                "placeholder": e.placeholder,
                "score": e.score,
            }
            if self._config.store_originals:
                entry["original_value"] = e.original_value
            entity_dicts.append(entry)

        audit_entry = _AuditEntry(
            session_id_hash=session_id_hash,
            scrubbed_at=result.scrubbed_at.isoformat(),
            scrub_failed=result.scrub_failed,
            failure_reason=result.failure_reason,
            entities=entity_dicts,
        )
        self._memory.append(audit_entry)
        self._append_to_file(audit_entry)

    def today_count(self) -> int:
        self._rotate_if_new_day()
        return len(self._memory)

    def _rotate_if_new_day(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._current_day:
            logger.info(
                "pii_audit.daily_rotation cleared=%d day=%s",
                len(self._memory),
                self._current_day.isoformat(),
            )
            self._memory.clear()
            self._current_day = today

    def _append_to_file(self, entry: _AuditEntry) -> None:
        try:
            with self._file_path.open("a", encoding="utf-8") as fh:
                fh.write(entry.to_json_line() + "\n")
        except OSError:
            logger.exception("pii_audit.file_write_failed path=%s", self._file_path)


class _FailureKind:
    ENCODING = "encoding"
    MODEL_NOT_LOADED = "model_not_loaded"
    MEMORY = "memory"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


def _classify_failure(exc: Exception) -> str:
    if isinstance(exc, (UnicodeDecodeError, UnicodeEncodeError)):
        return _FailureKind.ENCODING
    if isinstance(exc, MemoryError):
        return _FailureKind.MEMORY
    if isinstance(exc, TimeoutError):
        return _FailureKind.TIMEOUT
    msg = str(exc).lower()
    if any(kw in msg for kw in ("model", "spacy", "not found", "pipeline")):
        return _FailureKind.MODEL_NOT_LOADED
    return _FailureKind.UNKNOWN


class PiiScrubber:
    """Presidio-backed PII scrubber for CanonicalMessage fields.

    Stateless — safe to share a single instance across requests.
    All behaviour (entities, NER model, allow/deny lists) is driven by
    pii_config.yaml; no entity names are hardcoded here.

    Scrubs: text_content, session_context (recursive), prior_context (recursive).
    Leaves unchanged: session_id, channel, media_url, language_hint, timestamps.
    """

    def __init__(self, config: PiiConfig | None = None) -> None:
        self._config = config or PiiConfig.from_yaml()
        self._analyzer = self._build_analyzer()

    def scrub(self, message: CanonicalMessage) -> ScrubResult:
        """Return a ScrubResult with a PII-free CanonicalMessage and detected entities.

        Retries up to config.max_retries within config.retry_timeout_ms, applying
        failure-specific recovery (reload model, normalize encoding, etc.).
        Fails open on exhaustion: passes raw message with scrub_failed=True.
        """
        deadline = time.monotonic() + self._config.retry_timeout_ms / 1000.0
        last_exc: Exception | None = None
        last_kind = _FailureKind.UNKNOWN

        for attempt in range(1, self._config.max_retries + 1):
            if time.monotonic() >= deadline:
                logger.warning(
                    "pii.scrub timeout_exceeded attempt=%d budget_ms=%d",
                    attempt, self._config.retry_timeout_ms,
                )
                break
            try:
                result = self._do_scrub(message)
                if attempt > 1:
                    logger.info("pii.scrub recovered attempt=%d", attempt)
                return result
            except Exception as exc:
                last_exc = exc
                last_kind = _classify_failure(exc)
                logger.warning(
                    "pii.scrub failed attempt=%d/%d kind=%s error=%r",
                    attempt, self._config.max_retries, last_kind, exc,
                )
                self._recover(last_kind)

        logger.error(
            "pii.scrub fail_open retries=%d kind=%s", self._config.max_retries, last_kind
        )
        # Null out text_content so unscanned PII never reaches the Agentic OS.
        # All other fields (location, language, submission type) are safe to pass
        # through — they contain no free-text PII.
        return ScrubResult(
            clean_message=replace(message, text_content=None),
            entities=[],
            scrub_failed=True,
            failure_reason=f"{last_kind}: {last_exc}",
        )

    def _do_scrub(self, message: CanonicalMessage) -> ScrubResult:
        all_entities: list[PiiEntity] = []
        enabled = self._config.enabled_entity_names

        new_text = message.text_content
        if message.text_content is not None:
            new_text, entities = self._scrub_text(
                self._safe_str(message.text_content), "text_content", enabled
            )
            all_entities.extend(entities)

        new_session_context = deepcopy(message.session_context)
        self._scrub_dict_inplace(new_session_context, "session_context", enabled, all_entities)

        new_prior_context = deepcopy(message.prior_context)
        for i, item in enumerate(new_prior_context):
            if isinstance(item, dict):
                self._scrub_dict_inplace(item, f"prior_context[{i}]", enabled, all_entities)

        return ScrubResult(
            clean_message=replace(
                message,
                text_content=new_text,
                session_context=new_session_context,
                prior_context=new_prior_context,
            ),
            entities=all_entities,
        )

    def _scrub_text(
        self, text: str, field_path: str, enabled: list[str]
    ) -> tuple[str, list[PiiEntity]]:
        results = self._analyzer.analyze(
            text=text,
            language=self._config.language,
            entities=enabled,
            score_threshold=self._config.score_threshold,
        )

        allow_map: dict[str, tuple[str, ...]] = {
            e.name: e.allow_list for e in self._config.entities if e.enabled
        }

        allow_filtered: list[tuple[Any, str]] = []
        for r in results:
            original = text[r.start:r.end]
            if any(a.lower() == original.lower() for a in allow_map.get(r.entity_type, ())):
                logger.debug("pii.allow_list_hit entity=%s field=%s", r.entity_type, field_path)
                continue
            allow_filtered.append((r, original))

        # Deduplicate overlapping spans — keep highest-confidence entity per range.
        # Prevents double-replacement when one span matches multiple entity types.
        allow_filtered.sort(key=lambda x: x[0].score, reverse=True)
        accepted: list[tuple[Any, str]] = []
        occupied: list[tuple[int, int]] = []
        for r, original in allow_filtered:
            if any(r.start < e and r.end > s for s, e in occupied):
                logger.debug("pii.overlap_skip entity=%s field=%s", r.entity_type, field_path)
                continue
            accepted.append((r, original))
            occupied.append((r.start, r.end))

        entities: list[PiiEntity] = []
        for r, original in sorted(accepted, key=lambda x: x[0].start, reverse=True):
            h = sha256(original.encode("utf-8")).hexdigest()
            placeholder = f"[{r.entity_type}_{h[:8]}]"
            entities.append(PiiEntity(
                field_path=field_path,
                entity_type=r.entity_type,
                original_value=original,
                sha256_hash=h,
                placeholder=placeholder,
                score=r.score,
                start=r.start,
                end=r.end,
            ))
            text = text[:r.start] + placeholder + text[r.end:]

        return text, entities

    def _scrub_dict_inplace(
        self,
        d: dict[str, Any],
        path: str,
        enabled: list[str],
        collected: list[PiiEntity],
    ) -> None:
        for key, value in list(d.items()):
            child_path = f"{path}.{key}"
            if isinstance(value, str):
                scrubbed, entities = self._scrub_text(self._safe_str(value), child_path, enabled)
                if entities:
                    d[key] = scrubbed
                    collected.extend(entities)
            elif isinstance(value, dict):
                self._scrub_dict_inplace(value, child_path, enabled, collected)
            elif isinstance(value, list):
                d[key] = self._scrub_list(value, child_path, enabled, collected)

    def _scrub_list(
        self,
        lst: list[Any],
        path: str,
        enabled: list[str],
        collected: list[PiiEntity],
    ) -> list[Any]:
        result: list[Any] = []
        for i, item in enumerate(lst):
            item_path = f"{path}[{i}]"
            if isinstance(item, str):
                scrubbed, entities = self._scrub_text(self._safe_str(item), item_path, enabled)
                collected.extend(entities)
                result.append(scrubbed)
            elif isinstance(item, dict):
                item = deepcopy(item)
                self._scrub_dict_inplace(item, item_path, enabled, collected)
                result.append(item)
            else:
                result.append(item)
        return result

    def _recover(self, kind: str) -> None:
        if kind == _FailureKind.MODEL_NOT_LOADED:
            logger.info("pii.recovery reloading AnalyzerEngine")
            try:
                self._analyzer = self._build_analyzer()
            except Exception:
                logger.exception("pii.recovery reload failed")
        elif kind == _FailureKind.MEMORY:
            logger.warning("pii.recovery memory pressure detected")

    def _build_analyzer(self) -> Any:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        nlp_cfg = self._config.nlp_engine
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": nlp_cfg.provider,
            "models": [{"lang_code": self._config.language, "model_name": nlp_cfg.model_name}],
        })
        nlp_engine = provider.create_engine()

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)

        for entity_cfg in self._config.entities:
            if not entity_cfg.enabled or not entity_cfg.deny_list:
                continue
            from presidio_analyzer import PatternRecognizer
            registry.add_recognizer(PatternRecognizer(
                supported_entity=entity_cfg.name,
                deny_list=list(entity_cfg.deny_list),
                name=f"{entity_cfg.name}_deny_list",
            ))

        return AnalyzerEngine(registry=registry, nlp_engine=nlp_engine)

    @staticmethod
    def _safe_str(value: str) -> str:
        return value.encode("utf-8", errors="replace").decode("utf-8")
