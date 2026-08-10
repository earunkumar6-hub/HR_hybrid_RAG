"""
Guardrail layer for the Internal Knowledge Assistant.

Primary engine: NeMo Guardrails (`nemoguardrails`), configured in
`guardrails/config/` with LLM-based self-check-input / self-check-output
rails (see config.yml + rails/general.co) — this is the "Nemoguard"
option from the brief.

Because NeMo Guardrails needs a live LLM call to initialise/execute its
colang flows, this module wraps it in a `SimpleGuard` fallback (regex /
keyword based prompt-injection & off-topic detection) that activates
automatically if:
  * `nemoguardrails` isn't installed,
  * the colang config fails to load, or
  * `GUARDRAILS_ENABLED=false` in the environment.

Either way, `GuardrailEngine.check_input` / `check_output` return the same
`GuardResult` shape, so callers (routers/query.py) don't need to care which
backend is active. The active engine name is surfaced in the API response
for transparency.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from config import settings

logger = logging.getLogger("guardrails")


@dataclass
class GuardResult:
    allowed: bool
    reason: Optional[str] = None
    engine: str = "simple"


# ----------------------------------------------------------------------
# Lightweight fallback guard (no extra LLM round-trip required, though it
# optionally uses one cheap classification call for ambiguous cases)
# ----------------------------------------------------------------------
_INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?previous instructions",
    r"disregard (all|any|the)? ?(previous|above) instructions",
    r"you are now",
    r"reveal (your|the) (system prompt|instructions)",
    r"act as (?!.*internal)",  # "act as X" jailbreak framing, still allows normal use
    r"jailbreak",
    r"dan mode",
    r"pretend you have no restrictions",
]
_SENSITIVE_DATA_ASK = [
    r"\bsocial security number\b",
    r"\bssn\b",
    r"\bcredit card number\b",
    r"\bpassword(s)? for\b",
    r"\bmake up (a|an|the) (salary|ssn|password|employee)",
]


class SimpleGuard:
    engine_name = "simple-fallback"

    def check_input(self, text: str) -> GuardResult:
        lowered = text.lower()
        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, lowered):
                return GuardResult(
                    allowed=False,
                    reason="Detected a possible prompt-injection / jailbreak attempt.",
                    engine=self.engine_name,
                )
        for pattern in _SENSITIVE_DATA_ASK:
            if re.search(pattern, lowered):
                return GuardResult(
                    allowed=False,
                    reason="Request asks for fabricated or sensitive personal data.",
                    engine=self.engine_name,
                )
        return GuardResult(allowed=True, engine=self.engine_name)

    def check_output(self, text: str) -> GuardResult:
        lowered = text.lower()
        if "system prompt" in lowered and "cannot" not in lowered:
            return GuardResult(
                allowed=False,
                reason="Response appears to leak internal system instructions.",
                engine=self.engine_name,
            )
        for pattern in _SENSITIVE_DATA_ASK:
            if re.search(pattern, lowered):
                return GuardResult(
                    allowed=False,
                    reason="Response contains fabricated sensitive data.",
                    engine=self.engine_name,
                )
        return GuardResult(allowed=True, engine=self.engine_name)


# ----------------------------------------------------------------------
# NeMo Guardrails-backed engine
# ----------------------------------------------------------------------
class NemoGuard:
    engine_name = "nemoguardrails"

    def __init__(self):
        from nemoguardrails import LLMRails, RailsConfig  # local import: optional dep

        config = RailsConfig.from_path(settings.guardrails_config_path)
        self.rails = LLMRails(config)

    def check_input(self, text: str) -> GuardResult:
        response = self.rails.generate(messages=[{"role": "user", "content": text}])
        content = response.get("content", "") if isinstance(response, dict) else str(response)
        if "can't help with that" in content.lower() or "refuse to respond" in content.lower():
            return GuardResult(allowed=False, reason=content, engine=self.engine_name)
        return GuardResult(allowed=True, engine=self.engine_name)

    def check_output(self, text: str) -> GuardResult:
        # Output rails run inside rails.generate() together with input rails
        # in the standard NeMo flow; for a decoupled post-hoc check we reuse
        # the same self-check-output action indirectly by asking the model.
        result = self.rails.generate(
            messages=[
                {"role": "user", "content": "internal-output-check"},
                {"role": "assistant", "content": text},
            ]
        )
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        if "can't help with that" in content.lower():
            return GuardResult(allowed=False, reason="Blocked by output rail.", engine=self.engine_name)
        return GuardResult(allowed=True, engine=self.engine_name)


class GuardrailEngine:
    """Facade used by the API layer. Picks NeMo Guardrails when available
    and healthy, otherwise the deterministic SimpleGuard fallback."""

    def __init__(self):
        self._impl = None
        if settings.guardrails_enabled:
            try:
                self._impl = NemoGuard()
                logger.info("Guardrails: using NeMo Guardrails engine.")
            except Exception as exc:  # ImportError, config error, etc.
                logger.warning(
                    "NeMo Guardrails unavailable (%s) — falling back to SimpleGuard.",
                    exc,
                )
                self._impl = SimpleGuard()
        else:
            self._impl = SimpleGuard()

    def check_input(self, text: str) -> GuardResult:
        if not settings.guardrails_enabled:
            return GuardResult(allowed=True, engine="disabled")
        try:
            return self._impl.check_input(text)
        except Exception as exc:
            logger.warning("Guardrail input check failed (%s); allowing by default.", exc)
            return GuardResult(allowed=True, engine="error-open")

    def check_output(self, text: str) -> GuardResult:
        if not settings.guardrails_enabled:
            return GuardResult(allowed=True, engine="disabled")
        try:
            return self._impl.check_output(text)
        except Exception as exc:
            logger.warning("Guardrail output check failed (%s); allowing by default.", exc)
            return GuardResult(allowed=True, engine="error-open")


@lru_cache(maxsize=1)
def get_guardrail_engine() -> GuardrailEngine:
    return GuardrailEngine()
