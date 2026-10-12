"""Responses API calls, strict labels, bounded retries, and token probabilities."""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

from .benchmarks import BENCHMARKS
from .utils import MODEL, PROJECT_ROOT, utc_now

# Observed API rounding: positive logprobs <= 4e-6 and mass overshoot <= 6.5e-6.
# Preserve raw exp(logprob) mass; only tolerate this explicit numerical margin.
LOGPROB_NUMERIC_TOLERANCE = 1e-5


def load_environment():
    # Resolve the project's .env even when launched from a different directory.
    # Existing environment variables retain python-dotenv's usual precedence.
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY not found. Add it to the project's .env file.")
    return True


def make_client(timeout=60.0):
    load_environment()
    client = OpenAI()
    # Disable SDK retries so the explicit retry loop can count every attempt.
    return client.with_options(max_retries=0, timeout=timeout)


def parse_label(text, levels):
    """Accept only an entire numeric label, allowing surrounding whitespace.

    Unlike the notebook's first-digit regex, prose such as 'I chose 1' is invalid.
    This avoids turning malformed output into an arbitrary valid label.
    """
    if not isinstance(text, str):
        return None
    return {str(k): k for k in levels}.get(text.strip())


def token_text(candidate):
    token = candidate.get("token")
    if isinstance(token, str):
        return token
    try:
        return bytes(candidate.get("bytes") or []).decode("utf-8")
    except (ValueError, UnicodeDecodeError, TypeError):
        return ""


def extract_logprobs(payload, levels, min_valid_mass=0.90):
    """Distribution at the FIRST output-text token position only.

    Never shift to a later digit after a rationale/whitespace: that would silently
    estimate a different, prefix-conditional distribution. Distinct token variants
    for the same label are summed. Unseen labels have zero *observed* mass; their
    true probabilities outside top-20 are unknown. No smoothing is applied.
    """
    positions = []
    for output in payload.get("output", []):
        if output.get("type") == "message":
            for part in output.get("content", []):
                if part.get("type") == "output_text":
                    positions.extend(part.get("logprobs") or [])
    diagnostics = {"status": "missing_logprobs", "token_position": 0,
                   "output_token_count": len(positions), "first_token": None,
                   "top_logprobs": [], "n_returned_candidates": 0,
                   "numeric_tolerance": LOGPROB_NUMERIC_TOLERANCE,
                   "numeric_roundoff_detected": False,
                   "missing_labels": list(levels)}
    result = {"logprob_var": None, "valid_label_prob_mass": 0.0,
              "label_probabilities": {}, "logprob_diagnostics": diagnostics}
    if not positions:
        return result
    first = positions[0]
    candidates = first.get("top_logprobs") or []
    diagnostics.update(first_token=token_text(first), top_logprobs=candidates,
                       n_returned_candidates=len(candidates))
    mass_by_label = {k: 0.0 for k in levels}
    seen = set()
    for candidate in candidates:
        token = token_text(candidate)
        # A repeated raw token must not be counted twice; '1' and ' 1' differ.
        if token in seen:
            continue
        seen.add(token)
        label = parse_label(token, levels)
        lp = candidate.get("logprob")
        if label is None:
            continue
        if (not isinstance(lp, (int, float)) or not math.isfinite(lp)
                or lp > LOGPROB_NUMERIC_TOLERANCE):
            diagnostics["status"] = "invalid_logprob"
            return result
        if lp > 0:
            diagnostics["numeric_roundoff_detected"] = True
        mass_by_label[label] += math.exp(lp)
    mass = math.fsum(mass_by_label.values())
    result["valid_label_prob_mass"] = mass
    diagnostics["missing_labels"] = [k for k, p in mass_by_label.items() if p == 0]
    diagnostics["observed_label_mass"] = mass_by_label
    if mass > 1:
        diagnostics["numeric_roundoff_detected"] = True
    if mass == 0:
        diagnostics["status"] = "no_valid_label_tokens"
    elif mass > 1 + LOGPROB_NUMERIC_TOLERANCE:
        diagnostics["status"] = "invalid_probability_mass"
    elif mass < min_valid_mass:
        diagnostics["status"] = "low_valid_label_mass"
    else:
        probabilities = {k: p / mass for k, p in mass_by_label.items()}
        mean = math.fsum(k * p for k, p in probabilities.items())
        result["label_probabilities"] = probabilities
        result["logprob_var"] = math.fsum(p * (k - mean) ** 2
                                          for k, p in probabilities.items())
        diagnostics["status"] = "ok_partial_label_support" if diagnostics["missing_labels"] else "ok"
        if diagnostics["numeric_roundoff_detected"]:
            diagnostics["status"] += "_numeric_roundoff"
    return result


@dataclass
class CallFailure(Exception):
    """Sanitized failure; never persist/print an SDK exception body or headers."""
    kind: str
    status_code: int | None = None
    fatal: bool = False

    def __str__(self):
        status = f", HTTP {self.status_code}" if self.status_code else ""
        return f"{self.kind}{status}; inspect API access/configuration or retry later."


class LunaAnnotator:
    def __init__(self, client=None, *, timeout=60.0, max_attempts=5,
                 min_valid_mass=0.90, max_output_tokens=16):
        if max_attempts < 1 or not 0 < min_valid_mass <= 1 or max_output_tokens < 16:
            raise ValueError("Require attempts >= 1, 0 < min mass <= 1, output tokens >= 16.")
        self.client = client if client is not None else make_client(timeout)
        self.max_attempts = max_attempts
        self.min_valid_mass = min_valid_mass
        self.max_output_tokens = max_output_tokens

    def annotate_once(self, text, temperature, benchmark, *, logprobs=False,
                      before_attempt=None, after_error=None, stop_event=None):
        spec = BENCHMARKS[benchmark]
        params = dict(model=MODEL,
                      input=[{"role": "system", "content": spec["ann"]},
                             {"role": "user", "content": f"ITEM:\n{text}"}],
                      temperature=temperature, reasoning={"effort": "none"},
                      max_output_tokens=self.max_output_tokens, store=False)
        if logprobs:
            params.update(top_logprobs=20, include=["message.output_text.logprobs"])
        for attempt in range(self.max_attempts):
            if stop_event is not None and stop_event.is_set():
                raise CallFailure("interrupted")
            if before_attempt:
                before_attempt()
            try:
                response = self.client.responses.create(**params)
                break
            except (APIConnectionError, APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                transient = isinstance(exc, APIConnectionError) or status in (408, 409, 429) or (
                    status is not None and status >= 500)
                # Quota exhaustion is not a short-lived rate limit.
                body = getattr(exc, "body", None)
                code = body.get("code") if isinstance(body, dict) else None
                if isinstance(body, dict) and isinstance(body.get("error"), dict):
                    code = body["error"].get("code", code)
                if code == "insufficient_quota":
                    transient = False
                failure = CallFailure(type(exc).__name__, status, fatal=not transient)
                if after_error:
                    after_error(str(failure))
                if not transient or attempt == self.max_attempts - 1:
                    raise failure from None
                delay = min(30.0, 2 ** attempt + random.uniform(0, 0.5))
                headers = getattr(getattr(exc, "response", None), "headers", {})
                try:
                    delay = min(60.0, max(delay, float(headers.get("retry-after", 0))))
                except (ValueError, TypeError):
                    pass
                if stop_event is not None:
                    stop_event.wait(delay)
                else:
                    time.sleep(delay)
        payload = response if isinstance(response, dict) else response.model_dump(mode="json")
        parts = [part for out in payload.get("output", []) if out.get("type") == "message"
                 for part in out.get("content", []) if part.get("type") == "output_text"]
        raw = "".join(part.get("text", "") for part in parts)
        status = payload.get("status")
        label = parse_label(raw, spec["levels"]) if status == "completed" else None
        usage = payload.get("usage") or {}
        result = {"label": label, "raw_text": raw, "response_id": payload.get("id"),
                  "response_model": payload.get("model"), "response_status": status,
                  "incomplete_details": payload.get("incomplete_details"),
                  "created_at": utc_now(), "temperature": temperature,
                  "logprobs_requested": logprobs, "reasoning_effort": "none",
                  "input_tokens": usage.get("input_tokens", 0),
                  "output_tokens": usage.get("output_tokens", 0),
                  "cached_input_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)}
        if logprobs:
            result.update(extract_logprobs(payload, spec["levels"], self.min_valid_mass))
            # Whitelisted response structure for empirical inspection, with no input,
            # request headers, credentials, or repeatedly stored annotation prompt.
            result["output_structure"] = payload.get("output", [])
        return result
