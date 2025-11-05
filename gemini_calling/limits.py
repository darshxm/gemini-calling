from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple


class Limits:
    def __init__(self, rpm: Optional[int], tpm: Optional[int], rpd: Optional[int]):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd


# Default free-tier limits provided by user (common text-out models)
FREE_LIMITS: Dict[str, Limits] = {
    # Text-out models
    "gemini-2.5-pro": Limits(rpm=2, tpm=125_000, rpd=50),
    "gemini-2.5-flash-lite-preview": Limits(rpm=15, tpm=250_000, rpd=1_000),
    "gemini-2.5-flash-lite": Limits(rpm=15, tpm=250_000, rpd=1_000),
    "gemini-2.5-flash-preview": Limits(rpm=10, tpm=250_000, rpd=250),
    "gemini-2.5-flash": Limits(rpm=10, tpm=250_000, rpd=250),
    "gemini-2.0-flash-lite": Limits(rpm=30, tpm=1_000_000, rpd=200),
    "gemini-2.0-flash": Limits(rpm=15, tpm=1_000_000, rpd=200),
    # Other models
    "gemma-3n": Limits(rpm=30, tpm=15_000, rpd=14_400),
    "gemma-3": Limits(rpm=30, tpm=15_000, rpd=14_400),
    # Add more as needed. Models not listed have no client-side throttle by default.
}

# Paid Tier 1 limits provided by user (subset focused on text-out + some others)
TIER1_LIMITS: Dict[str, Limits] = {
    # Text-out models
    "gemini-2.5-pro": Limits(rpm=150, tpm=2_000_000, rpd=10_000),
    "gemini-2.5-flash-lite-preview": Limits(rpm=4_000, tpm=4_000_000, rpd=None),
    "gemini-2.5-flash-lite": Limits(rpm=4_000, tpm=4_000_000, rpd=None),
    "gemini-2.5-flash-preview": Limits(rpm=1_000, tpm=1_000_000, rpd=10_000),
    "gemini-2.5-flash": Limits(rpm=1_000, tpm=1_000_000, rpd=10_000),
    "gemini-2.0-flash-lite": Limits(rpm=4_000, tpm=4_000_000, rpd=None),
    "gemini-2.0-flash": Limits(rpm=2_000, tpm=4_000_000, rpd=None),
    # Other models (subset)
    "gemma-3n": Limits(rpm=30, tpm=15_000, rpd=14_400),
    "gemma-3": Limits(rpm=30, tpm=15_000, rpd=14_400),
}


def resolve_limits(model: str, overrides: Optional[Dict[str, Limits]] = None) -> Optional[Limits]:
    """Return Limits for a model using longest-prefix match among overrides or defaults."""
    model_l = model.lower()
    # Base table is empty; caller supplies chosen tier as overrides or merges presets below
    table: Dict[str, Limits] = {}
    if overrides:
        table.update({k.lower(): v for k, v in overrides.items()})
    best_key = None
    for key in table.keys():
        if model_l.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return table.get(best_key) if best_key else None


class RateLimiter:
    """Simple per-key, per-model limiter for RPM/RPD and TPM (token bucket per minute)."""

    def __init__(self, limits: Optional[Dict[str, Limits]] = None):
        # limits dict is model-prefix -> Limits (already tier-resolved)
        self._overrides = {k.lower(): v for k, v in (limits or {}).items()}
        # State: (key, model) -> windows/bucket
        self._req_1m: Dict[Tuple[str, str], Deque[float]] = {}
        self._req_1d: Dict[Tuple[str, str], Deque[float]] = {}
        self._bucket: Dict[Tuple[str, str], Dict[str, float]] = {}
        # Global state: (GLOBAL, model)
        self._req_1m_g: Dict[str, Deque[float]] = {}
        self._req_1d_g: Dict[str, Deque[float]] = {}
        self._bucket_g: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _now() -> float:
        return time.time()

    def _get_limits(self, model: str) -> Optional[Limits]:
        return resolve_limits(model, overrides=self._overrides)

    def _prune(self, key: str, model: str, now: float) -> None:
        k = (key, model)
        if k in self._req_1m:
            dq = self._req_1m[k]
            window = now - 60.0
            while dq and dq[0] <= window:
                dq.popleft()
        if k in self._req_1d:
            dq = self._req_1d[k]
            window = now - 86400.0
            while dq and dq[0] <= window:
                dq.popleft()
        if model in self._req_1m_g:
            dq = self._req_1m_g[model]
            window = now - 60.0
            while dq and dq[0] <= window:
                dq.popleft()
        if model in self._req_1d_g:
            dq = self._req_1d_g[model]
            window = now - 86400.0
            while dq and dq[0] <= window:
                dq.popleft()

    def _ensure_structs(self, key: str, model: str) -> None:
        k = (key, model)
        self._req_1m.setdefault(k, deque())
        self._req_1d.setdefault(k, deque())
        self._bucket.setdefault(k, {"avail": 0.0, "last": self._now(), "cap": 0.0, "rate": 0.0})
        self._req_1m_g.setdefault(model, deque())
        self._req_1d_g.setdefault(model, deque())
        self._bucket_g.setdefault(model, {"avail": 0.0, "last": self._now(), "cap": 0.0, "rate": 0.0})

    def _refill_bucket(self, key: str, model: str, now: float, tpm: int) -> None:
        k = (key, model)
        b = self._bucket[k]
        rate = tpm / 60.0  # tokens per second
        cap = float(tpm)
        last = b.get("last", now)
        avail = b.get("avail", 0.0)
        avail = min(cap, avail + max(0.0, now - last) * rate)
        b.update({"avail": avail, "last": now, "cap": cap, "rate": rate})

        # Global bucket
        bg = self._bucket_g[model]
        rate_g = tpm / 60.0
        cap_g = float(tpm)
        last_g = bg.get("last", now)
        avail_g = bg.get("avail", 0.0)
        avail_g = min(cap_g, avail_g + max(0.0, now - last_g) * rate_g)
        bg.update({"avail": avail_g, "last": now, "cap": cap_g, "rate": rate_g})

    def get_wait_time(self, key: str, model: str, needed_tokens: int) -> float:
        """Return seconds to wait to satisfy per-key AND global limits. 0 if ready now."""
        lim = self._get_limits(model)
        if not lim:
            return 0.0

        self._ensure_structs(key, model)
        now = self._now()
        self._prune(key, model, now)

        wait_times = [0.0]

        # Per-key RPM
        if lim.rpm is not None and lim.rpm > 0:
            dq = self._req_1m[(key, model)]
            if len(dq) >= lim.rpm:
                oldest = dq[0]
                wait_times.append(max(0.0, oldest + 60.0 - now))

        # Per-key RPD
        if lim.rpd is not None and lim.rpd > 0:
            dq = self._req_1d[(key, model)]
            if len(dq) >= lim.rpd:
                oldest = dq[0]
                wait_times.append(max(0.0, oldest + 86400.0 - now))

        # Per-key TPM (token bucket)
        if lim.tpm is not None and lim.tpm > 0 and needed_tokens > 0:
            self._refill_bucket(key, model, now, lim.tpm)
            b = self._bucket[(key, model)]
            avail = b.get("avail", 0.0)
            if needed_tokens > avail:
                rate = b.get("rate", lim.tpm / 60.0)
                deficit = needed_tokens - avail
                wait_times.append(deficit / max(rate, 1e-9))

        # Global RPM
        if lim.rpm is not None and lim.rpm > 0:
            dq = self._req_1m_g[model]
            if len(dq) >= lim.rpm:
                oldest = dq[0]
                wait_times.append(max(0.0, oldest + 60.0 - now))

        # Global RPD
        if lim.rpd is not None and lim.rpd > 0:
            dq = self._req_1d_g[model]
            if len(dq) >= lim.rpd:
                oldest = dq[0]
                wait_times.append(max(0.0, oldest + 86400.0 - now))

        # Global TPM
        if lim.tpm is not None and lim.tpm > 0 and needed_tokens > 0:
            # Refill already updates global bucket too
            bg = self._bucket_g[model]
            avail_g = bg.get("avail", 0.0)
            if needed_tokens > avail_g:
                rate_g = bg.get("rate", lim.tpm / 60.0)
                deficit_g = needed_tokens - avail_g
                wait_times.append(deficit_g / max(rate_g, 1e-9))

        return max(wait_times)

    def consume(self, key: str, model: str, needed_tokens: int) -> None:
        lim = self._get_limits(model)
        if not lim:
            return
        self._ensure_structs(key, model)
        now = self._now()
        self._prune(key, model, now)
        # record requests
        if lim.rpm:
            self._req_1m[(key, model)].append(now)
            self._req_1m_g[model].append(now)
        if lim.rpd:
            self._req_1d[(key, model)].append(now)
            self._req_1d_g[model].append(now)
        # consume tokens
        if lim.tpm and needed_tokens > 0:
            self._refill_bucket(key, model, now, lim.tpm)
            b = self._bucket[(key, model)]
            b["avail"] = max(0.0, b.get("avail", 0.0) - float(needed_tokens))
            bg = self._bucket_g[model]
            bg["avail"] = max(0.0, bg.get("avail", 0.0) - float(needed_tokens))

# Convenience presets per tier
PRESETS = {
    "free": FREE_LIMITS,
    "tier1": TIER1_LIMITS,
}
