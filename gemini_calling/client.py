from __future__ import annotations

import json
import itertools
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import request, error

from .keys import load_keys, load_named_keys
from .limits import RateLimiter, PRESETS
from .keys import load_limits_tier


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiAPIError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"Gemini API error {status}: {body}")
        self.status = status
        self.body = body


class GenerateContentResult:
    """Simple wrapper for a generateContent response."""

    def __init__(self, raw: Dict[str, Any]):
        self.raw = raw
        # Best-effort extraction of top candidate text
        self.text = self._extract_text(raw)
        self.function_calls = self._extract_function_calls(raw)

    def parse_json(self) -> Optional[Any]:
        """Attempt to parse response.text as JSON; return None on failure."""
        try:
            return json.loads(self.text)
        except Exception:
            return None

    @staticmethod
    def _extract_text(raw: Dict[str, Any]) -> str:
        try:
            cands = raw.get("candidates") or []
            if not cands:
                return ""
            content = cands[0].get("content") or {}
            parts = content.get("parts") or []
            for p in parts:
                if "text" in p:
                    return p["text"] or ""
            return ""
        except Exception:
            return ""

    @staticmethod
    def _extract_function_calls(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        try:
            # Some responses may have a top-level functionCalls field
            top = raw.get("functionCalls") or raw.get("function_calls")
            if isinstance(top, list):
                for c in top:
                    name = c.get("name")
                    args = c.get("args")
                    if name:
                        calls.append({"name": name, "args": args})
            # Parse from candidates > content > parts
            cands = raw.get("candidates") or []
            for cand in cands:
                content = cand.get("content") or {}
                parts = content.get("parts") or []
                for p in parts:
                    fc = p.get("functionCall") or p.get("function_call")
                    if fc and isinstance(fc, dict):
                        name = fc.get("name")
                        args = fc.get("args")
                        if name:
                            calls.append({"name": name, "args": args})
        except Exception:
            pass
        return calls


class GeminiClient:
    def __init__(
        self,
        api_keys: Optional[Iterable[str]] = None,
        default_model: str = "gemini-2.5-flash",
        timeout: int = 60,
        base_url: str = GEMINI_BASE_URL,
        # Retry and backoff configuration
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        max_backoff: float = 30.0,
        retry_statuses: Tuple[int, ...] = (429, 500, 502, 503, 504),
        rotate_keys_on_retry: bool = True,
        respect_retry_after: bool = True,
        # Rate limiting
        enable_rate_limit: bool = True,
        limits_tier: Optional[str] = None,  # 'free' | 'tier1' | None
        model_limits: Optional[dict] = None,  # extra/override limits
        # Token estimation
        tokens_estimator: Optional[callable] = None,
        # Token-per-minute enforcement (TPM): when False, only RPM/RPD enforced
        enforce_tpm: bool = False,
    ) -> None:
        keys = list(api_keys) if api_keys is not None else load_keys()
        if not keys:
            # Allow usage with explicit per-call api_key override, but warn via error later if still missing
            keys = []
        self._keys: List[str] = keys
        self._rr = itertools.cycle(range(len(keys))) if keys else None
        self.default_model = default_model
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        # Retry config
        self.max_retries = max_retries
        self.backoff_factor = float(backoff_factor)
        self.max_backoff = float(max_backoff)
        self.retry_statuses = tuple(retry_statuses)
        self.rotate_keys_on_retry = bool(rotate_keys_on_retry)
        self.respect_retry_after = bool(respect_retry_after)
        # Per-key state: cooldowns and disable flags
        # key -> {"cooldown_until": ts(float) or 0.0, "disabled": bool, "failures": int}
        self._key_state: Dict[str, Dict[str, Any]] = {
            k: {"cooldown_until": 0.0, "disabled": False, "failures": 0} for k in keys
        }
        # Rate limiter
        self.enable_rate_limit = enable_rate_limit
        self.enforce_tpm = bool(enforce_tpm)
        # Determine tier from param or env/.env
        tier = (limits_tier or load_limits_tier() or "free").lower()
        preset = PRESETS.get(tier, PRESETS["free"]).copy()
        if model_limits:
            # merge/override
            preset.update({k.lower(): v for k, v in model_limits.items()})
        self._rate_limiter = RateLimiter(limits=preset)
        self._tokens_estimator = tokens_estimator

    def _is_key_available(self, key: str) -> bool:
        st = self._key_state.get(key)
        if not st:
            return True
        if st.get("disabled"):
            return False
        return time.time() >= float(st.get("cooldown_until", 0.0))

    def _next_available_key(self) -> Optional[str]:
        if not self._keys:
            return None
        if self._rr is None:
            # single key
            k = self._keys[0]
            return k if self._is_key_available(k) else None
        # probe up to N keys once
        start_idx = next(self._rr)
        for offset in range(len(self._keys)):
            idx = (start_idx + offset) % len(self._keys)
            k = self._keys[idx]
            if self._is_key_available(k):
                return k
        return None

    def _best_key_for_limits(self, model: str, needed_tokens: int) -> Tuple[Optional[str], float]:
        """Return the key with the smallest wait; (key, wait_seconds). Prefers immediate (wait=0)."""
        best_key = None
        best_wait = float("inf")
        for k in self._keys:
            if not self._is_key_available(k):
                continue
            wait = 0.0
            if self.enable_rate_limit:
                nt = needed_tokens if self.enforce_tpm else 0
                wait = float(self._rate_limiter.get_wait_time(k, model, nt))
            if wait == 0.0:
                return k, 0.0
            if wait < best_wait:
                best_wait = wait
                best_key = k
        return best_key, (0.0 if best_key is None else best_wait)

    def _pick_key(self, api_key: Optional[str]) -> str:
        if api_key:
            return api_key
        k = self._next_available_key()
        if k:
            return k
        # If none available, find earliest cooldown and sleep a little (bounded)
        if not self._keys:
            raise ValueError("No API key available. Set GEMINI_API_KEY(S), provide a .env, or pass api_key=")
        soonest = min(
            [self._key_state.get(k, {}).get("cooldown_until", 0.0) for k in self._keys]
        )
        wait = max(soonest - time.time(), 0.0)
        if wait > 0:
            time.sleep(min(wait, 2.0))  # small wait to allow cooldowns to expire
            k2 = self._next_available_key()
            if k2:
                return k2
        # Still none; raise
        raise ValueError("All API keys are cooling down or disabled. Try again later.")

    def _mark_success(self, key: str) -> None:
        st = self._key_state.get(key)
        if st:
            st["failures"] = 0
            st["cooldown_until"] = 0.0

    def _mark_failure(self, key: str, status: Optional[int], headers: Optional[Dict[str, Any]] = None, attempt: int = 0) -> None:
        st = self._key_state.setdefault(key, {"cooldown_until": 0.0, "disabled": False, "failures": 0})
        st["failures"] = int(st.get("failures", 0)) + 1
        # Auth errors: disable key
        if status in (401, 403):
            st["disabled"] = True
            st["cooldown_until"] = float("inf")
            return
        # Rate limit: use Retry-After when available
        cooldown = 0.0
        if status == 429 and self.respect_retry_after and headers:
            ra = headers.get("Retry-After") or headers.get("retry-after")
            if ra:
                try:
                    cooldown = float(ra)
                except Exception:
                    # Retry-After may be HTTP date; ignore for simplicity here
                    cooldown = 0.0
        # Exponential backoff fallback
        if cooldown <= 0.0:
            cooldown = min(self.max_backoff, self.backoff_factor * (2 ** max(attempt - 1, 0)))
        st["cooldown_until"] = time.time() + float(cooldown)

    # Helpers to build parts for multi-turn tool use
    @staticmethod
    def build_function_response_part(name: str, response: Dict[str, Any]) -> Dict[str, Any]:
        return {"functionResponse": {"name": name, "response": response}}

    @staticmethod
    def build_text_part(text: str) -> Dict[str, Any]:
        return {"text": text}


def gemini_request(
    # Basic
    prompt: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    contents: Optional[List[Dict[str, Any]]] = None,
    # Key selection (no secrets in code required)
    env_path: Optional[str] = None,
    key_index: Optional[int] = None,          # 1-based index into .env keys
    key_match: Optional[str] = None,          # substring match against keys
    key_name: Optional[str] = None,           # named key identifier (GEMINI_API_KEY_<NAME>)
    api_key: Optional[str] = None,            # direct override (not recommended)
    # Generation options
    system_instruction: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
    thinking_budget: Optional[int] = None,
    # Structured output
    response_mime_type: Optional[str] = None,
    response_schema: Optional[Dict[str, Any]] = None,
    response_json_schema: Optional[Dict[str, Any]] = None,
    response_enum: Optional[List[str]] = None,
    # Function calling
    tools: Optional[List[Dict[str, Any]]] = None,
    function_calling_mode: Optional[str] = None,
    allowed_function_names: Optional[List[str]] = None,
    tool_config: Optional[Dict[str, Any]] = None,
    # Client behavior
    timeout: int = 60,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    max_backoff: float = 30.0,
    retry_statuses: Tuple[int, ...] = (429, 500, 502, 503, 504),
    rotate_keys_on_retry: bool = True,
    respect_retry_after: bool = True,
    enable_rate_limit: bool = True,
    enforce_tpm: bool = False,
    limits_tier: Optional[str] = None,  # 'free' | 'tier1' | None to auto
    model_limits: Optional[dict] = None,
    tokens_estimator: Optional[callable] = None,
    # Token planning
    estimated_output_tokens: int = 0,
    # Extra config passthrough
    extra_generation_config: Optional[Dict[str, Any]] = None,
) -> GenerateContentResult:
    """One-shot Gemini request.

    - Loads keys from env/.env; select a specific key with key_index (1-based) or key_match.
    - Defaults to plain text output. Use structured output or tools via parameters.
    - Returns a GenerateContentResult with .text, .function_calls, and .raw.
    """

    # Load keys from env or .env (do not require embedding in code)
    keys = load_keys(path=env_path)
    named = load_named_keys(path=env_path)

    # Build client with provided behavior
    client = GeminiClient(
        api_keys=keys,
        default_model=model,
        timeout=timeout,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        max_backoff=max_backoff,
        retry_statuses=retry_statuses,
        rotate_keys_on_retry=rotate_keys_on_retry,
        respect_retry_after=respect_retry_after,
        enable_rate_limit=enable_rate_limit,
        enforce_tpm=enforce_tpm,
        limits_tier=limits_tier,
        model_limits=model_limits,
        tokens_estimator=tokens_estimator,
    )

    # Choose a specific key if requested
    selected_key: Optional[str] = None
    if api_key:
        selected_key = api_key
    elif key_name:
        # Try exact name, then case-insensitive
        if key_name in named:
            selected_key = named[key_name]
        else:
            kn = key_name.lower()
            for n, k in named.items():
                if n.lower() == kn:
                    selected_key = k
                    break
        if not selected_key:
            raise KeyError(f"Named key '{key_name}' not found in env/.env (GEMINI_API_KEY_<NAME>)")
    elif keys:
        if key_index is not None:
            # 1-based index for human-friendliness
            if key_index <= 0 or key_index > len(keys):
                raise IndexError(f"key_index out of range: 1..{len(keys)}")
            selected_key = keys[key_index - 1]
        elif key_match:
            km = key_match.strip()
            for k in keys:
                if km and km in k:
                    selected_key = k
                    break

    # Dispatch request
    return client.generate_text(
        prompt=prompt,
        model=model,
        api_key=selected_key,
        contents=contents,
        system_instruction=system_instruction,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop_sequences=stop_sequences,
        thinking_budget=thinking_budget,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
        response_json_schema=response_json_schema,
        response_enum=response_enum,
        tools=tools,
        function_calling_mode=function_calling_mode,
        allowed_function_names=allowed_function_names,
        tool_config=tool_config,
        extra_generation_config=extra_generation_config,
        estimated_output_tokens=estimated_output_tokens,
    )

    def key_health(self) -> List[Dict[str, Any]]:
        """Return current health info for managed keys (masked)."""
        def mask(k: str) -> str:
            return k[:4] + "..." + k[-4:] if len(k) > 8 else "****"

        out = []
        for k in self._keys:
            st = self._key_state.get(k, {})
            out.append({
                "key": mask(k),
                "disabled": bool(st.get("disabled", False)),
                "cooldown_until": float(st.get("cooldown_until", 0.0)),
                "failures": int(st.get("failures", 0)),
                "available": self._is_key_available(k),
            })
        return out

    def generate_text(
        self,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        thinking_budget: Optional[int] = None,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        response_json_schema: Optional[Dict[str, Any]] = None,
        response_enum: Optional[List[str]] = None,
        # Function calling
        tools: Optional[List[Dict[str, Any]]] = None,
        function_calling_mode: Optional[str] = None,  # 'AUTO'|'ANY'|'NONE'
        allowed_function_names: Optional[List[str]] = None,
        tool_config: Optional[Dict[str, Any]] = None,
        # For advanced use: pass raw contents directly (overrides prompt)
        contents: Optional[List[Dict[str, Any]]] = None,
        extra_generation_config: Optional[Dict[str, Any]] = None,
        estimated_output_tokens: int = 0,
    ) -> GenerateContentResult:
        """Call the Gemini generateContent REST endpoint and return a parsed result.

        Only text input is supported here. For multimodal or streaming, extend this client.
        """

        chosen_model = model or self.default_model
        url = f"{self.base_url}/models/{chosen_model}:generateContent"

        if (tools or function_calling_mode or tool_config) and (
            response_schema or response_json_schema or response_enum or (response_mime_type in ("application/json", "text/x.enum"))
        ):
            raise ValueError("Function calling and structured output cannot be used in the same call.")

        if contents is not None:
            body: Dict[str, Any] = {"contents": contents}
        else:
            if not prompt:
                raise ValueError("prompt is required when contents is not provided")
            body = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                        ]
                    }
                ]
            }

        gen_cfg: Dict[str, Any] = {}
        if temperature is not None:
            gen_cfg["temperature"] = float(temperature)
        if top_p is not None:
            gen_cfg["topP"] = float(top_p)
        if top_k is not None:
            gen_cfg["topK"] = int(top_k)
        if stop_sequences:
            gen_cfg["stopSequences"] = list(stop_sequences)
        # Structured output configuration
        if response_enum:
            # Convenience for enums; defaults MIME if user didn't set one
            if not response_mime_type:
                response_mime_type = "text/x.enum"
            # Build enum schema
            response_schema = {
                "type": "STRING",
                "enum": list(response_enum),
            }
        if response_mime_type:
            gen_cfg["responseMimeType"] = response_mime_type
        if response_schema:
            # Normalize type casing to API expectations (upper-case)
            def _norm(sch: Any) -> Any:
                if isinstance(sch, dict):
                    out = {}
                    for k, v in sch.items():
                        if k == "type" and isinstance(v, str):
                            out[k] = v.upper()
                        else:
                            out[k] = _norm(v)
                    return out
                elif isinstance(sch, list):
                    return [_norm(x) for x in sch]
                else:
                    return sch
            gen_cfg["responseSchema"] = _norm(response_schema)
        if response_json_schema:
            # Preview feature (snake_case field name per API doc)
            gen_cfg["response_json_schema"] = response_json_schema
        if thinking_budget is not None:
            gen_cfg.setdefault("thinkingConfig", {})["thinkingBudget"] = int(thinking_budget)
        if extra_generation_config:
            gen_cfg.update(extra_generation_config)

        if gen_cfg:
            body["generationConfig"] = gen_cfg

        if system_instruction:
            body["system_instruction"] = {
                "parts": [
                    {"text": system_instruction},
                ]
            }

        # Function calling config
        if tools:
            body["tools"] = tools
        if tool_config or function_calling_mode or allowed_function_names:
            fc_cfg: Dict[str, Any] = {}
            if function_calling_mode:
                fc_cfg["mode"] = function_calling_mode
            if allowed_function_names:
                fc_cfg["allowedFunctionNames"] = list(allowed_function_names)
            body["toolConfig"] = {"functionCallingConfig": fc_cfg}
            if tool_config:
                # Merge user-provided tool_config (takes precedence)
                body["toolConfig"].update(tool_config)

        data = json.dumps(body).encode("utf-8")

        # Estimate tokens for throttling
        if self._tokens_estimator:
            est_in = int(max(1, self._tokens_estimator(prompt or "")))
        else:
            # rough heuristic: 4 chars per token
            est_in = max(1, (len(prompt or "") + 3) // 4)
        need_tokens = max(0, int(est_in) + int(max(0, estimated_output_tokens)))

        attempts = 0
        last_exc: Optional[Exception] = None
        while True:
            attempts += 1
            # Choose the best key considering cooldowns and rate limits
            if api_key:
                # Respect user-provided key
                key = api_key
                if not self._is_key_available(key):
                    time.sleep(0.1)
                if self.enable_rate_limit:
                    nt = need_tokens if self.enforce_tpm else 0
                    wait = self._rate_limiter.get_wait_time(key, chosen_model, nt)
                    if wait > 0:
                        time.sleep(min(wait, 2.0))
            else:
                key = self._pick_key(None)
                if self.enable_rate_limit:
                    chosen_key, wait = self._best_key_for_limits(chosen_model, need_tokens if self.enforce_tpm else 0)
                    if chosen_key:
                        key = chosen_key
                    if wait > 0:
                        time.sleep(min(wait, 2.0))

            req = request.Request(url, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("x-goog-api-key", key)

            try:
                # Consume rate-limits at request start (requests and tokens)
                if self.enable_rate_limit:
                    self._rate_limiter.consume(key, chosen_model, need_tokens if self.enforce_tpm else 0)
                with request.urlopen(req, data=data, timeout=self.timeout) as resp:
                    status = resp.getcode()
                    resp_body = resp.read().decode("utf-8")
                    if status < 200 or status >= 300:
                        # Non-2xx statuses shouldn't happen here, but treat as HTTPError
                        self._mark_failure(key, status, resp.headers, attempts)
                        if status in self.retry_statuses and attempts <= self.max_retries:
                            if self.rotate_keys_on_retry:
                                # rotate by advancing rr by 1 (already advanced in _pick_key)
                                pass
                            continue
                        raise GeminiAPIError(status, resp_body)
                    payload = json.loads(resp_body or "{}")
                    self._mark_success(key)
                    return GenerateContentResult(payload)
            except error.HTTPError as e:
                # Read body once for diagnostics
                try:
                    err_body = e.read().decode("utf-8")
                except Exception:
                    err_body = str(e)

                status = getattr(e, 'code', None)
                headers = getattr(e, 'headers', None)
                self._mark_failure(key, status, headers, attempts)

                if status in (401, 403):
                    # Key disabled; try another immediately if possible
                    last_exc = GeminiAPIError(status, err_body)
                elif status in self.retry_statuses and attempts <= self.max_retries:
                    # backoff before next try if not rotating keys
                    if not self.rotate_keys_on_retry:
                        delay = min(self.max_backoff, self.backoff_factor * (2 ** max(attempts - 1, 0)))
                        time.sleep(delay)
                    last_exc = GeminiAPIError(status, err_body)
                else:
                    # Non-retryable
                    raise GeminiAPIError(status or 0, err_body)
            except error.URLError as e:
                # Network error; retry
                self._mark_failure(key, None, None, attempts)
                if attempts <= self.max_retries:
                    if not self.rotate_keys_on_retry:
                        delay = min(self.max_backoff, self.backoff_factor * (2 ** max(attempts - 1, 0)))
                        time.sleep(delay)
                    last_exc = RuntimeError(f"Network error: {e}")
                else:
                    raise RuntimeError(f"Network error: {e}")

            # If here, we will retry if allowed; otherwise break
            if attempts > self.max_retries:
                if last_exc:
                    raise last_exc
                raise RuntimeError("Max retries exceeded")
