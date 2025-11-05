from __future__ import annotations

import os
from typing import Iterable, List, Optional


ENV_SINGLE = "GEMINI_API_KEY"
ENV_MULTI = "GEMINI_API_KEYS"
ENV_LIMITS_TIER = "GEMINI_LIMITS_TIER"  # "free" | "tier1"
ENV_NAMED_PREFIX = "GEMINI_API_KEY_"    # e.g., GEMINI_API_KEY_MYFREE=abcd


def _parse_multi(s: str) -> List[str]:
    # Accept comma or newline separated keys. Strip spaces; drop empties.
    raw = [p.strip() for chunk in s.split("\n") for p in chunk.split(",")]
    return [k for k in raw if k]


def _read_env_file(path: str) -> dict:
    values: dict[str, str] = {}
    if not os.path.exists(path):
        return values
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                values[key] = val
    except Exception:
        # Be forgiving; return what we have
        pass
    return values


def _write_env_file(path: str, values: dict, header: Optional[str] = None) -> None:
    lines: List[str] = []
    if header:
        for hline in header.splitlines():
            lines.append(f"# {hline}")
    for k, v in values.items():
        lines.append(f"{k}={v}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def load_keys(path: Optional[str] = None) -> List[str]:
    """Load Gemini API keys from environment or an optional .env path.

    Order of precedence:
    - ENV (GEMINI_API_KEYS)
    - ENV (GEMINI_API_KEY)
    - .env file (path, or "./.env" if present)
    """
    # Environment variables first
    if ENV_MULTI in os.environ and os.environ[ENV_MULTI].strip():
        keys = _parse_multi(os.environ[ENV_MULTI])
        return list(dict.fromkeys(keys))  # dedupe, preserve order

    if ENV_SINGLE in os.environ and os.environ[ENV_SINGLE].strip():
        return [os.environ[ENV_SINGLE].strip()]

    # Fallback to .env if present
    search_path = path or (".env" if os.path.exists(".env") else None)
    if search_path:
        vals = _read_env_file(search_path)
        if ENV_MULTI in vals and vals[ENV_MULTI].strip():
            keys = _parse_multi(vals[ENV_MULTI])
            return list(dict.fromkeys(keys))
        if ENV_SINGLE in vals and vals[ENV_SINGLE].strip():
            return [vals[ENV_SINGLE].strip()]

    return []


def load_named_keys(path: Optional[str] = None) -> dict:
    """Load named Gemini API keys from environment or .env.

    Recognizes variables with prefix GEMINI_API_KEY_<NAME>=<key>.
    Returns a dict mapping the given NAME (as-is) to the key.
    """
    named: dict[str, str] = {}

    # From environment
    for k, v in os.environ.items():
        if not k.upper().startswith(ENV_NAMED_PREFIX):
            continue
        name = k[len(ENV_NAMED_PREFIX) :]
        if v.strip():
            named[name] = v.strip()

    # From .env
    search_path = path or (".env" if os.path.exists(".env") else None)
    if search_path:
        vals = _read_env_file(search_path)
        for k, v in vals.items():
            if not k.upper().startswith(ENV_NAMED_PREFIX):
                continue
            name = k[len(ENV_NAMED_PREFIX) :]
            if v.strip():
                # env has precedence; otherwise fill from file
                named.setdefault(name, v.strip())

    return named


def load_limits_tier(path: Optional[str] = None) -> Optional[str]:
    """Load limits tier from env or .env. Returns 'free', 'tier1', or None."""
    v = os.environ.get(ENV_LIMITS_TIER)
    if v:
        v = v.strip().lower()
        if v in ("free", "tier1"):
            return v
    search_path = path or (".env" if os.path.exists(".env") else None)
    if search_path:
        vals = _read_env_file(search_path)
        v = vals.get(ENV_LIMITS_TIER, "").strip().lower()
        if v in ("free", "tier1"):
            return v
    return None


def set_keys(keys: Iterable[str], path: str = ".env") -> None:
    """Write the provided keys into a .env file.

    - Uses GEMINI_API_KEYS with comma-separated values.
    - Overwrites existing file content.
    """
    key_list = [k.strip() for k in keys if k and k.strip()]
    if not key_list:
        raise ValueError("No valid keys provided")
    values = {ENV_MULTI: ",".join(key_list)}
    header = (
        "Gemini API keys.\n"
        f"Set either {ENV_MULTI} (comma or newline separated) or {ENV_SINGLE}.\n"
        "This file is not automatically loaded by Python; use tools like python-dotenv\n"
        "or load via gemini_calling.keys.load_keys()."
    )
    _write_env_file(path, values, header=header)


def create_env(keys: Optional[Iterable[str]] = None, path: str = ".env", overwrite: bool = False) -> str:
    """Create a .env file with placeholders or provided keys.

    Returns the written path. Will not overwrite unless explicitly allowed.
    """
    if os.path.exists(path) and not overwrite:
        return path

    if keys:
        set_keys(keys, path)
        return path

    values = {
        ENV_MULTI: "",  # fill in as: key1,key2,key3
        ENV_SINGLE: "",  # or set a single key here
        ENV_LIMITS_TIER: "free",  # or 'tier1'
    }
    header = (
        "Template .env for Gemini API keys.\n"
        f"Fill {ENV_MULTI} or {ENV_SINGLE}. If both are set, {ENV_MULTI} takes precedence.\n"
        f"Set {ENV_LIMITS_TIER} to 'free' or 'tier1' to control client-side throttling."
    )
    _write_env_file(path, values, header=header)
    return path


def set_limits_tier(tier: str, path: str = ".env") -> None:
    tier_l = (tier or "").strip().lower()
    if tier_l not in ("free", "tier1"):
        raise ValueError("tier must be 'free' or 'tier1'")
    existing = _read_env_file(path) if os.path.exists(path) else {}
    existing[ENV_LIMITS_TIER] = tier_l
    _write_env_file(path, existing, header=f"Update {ENV_LIMITS_TIER}")


def configure_limits_interactive(path: str = ".env") -> str:
    """Ask user for limits tier and write to .env. Returns chosen tier."""
    print("Select your Gemini API tier:")
    print("  1) Free")
    print("  2) Paid Tier 1")
    sel = input("Enter 1 or 2 [1]: ").strip() or "1"
    tier = "tier1" if sel == "2" else "free"
    set_limits_tier(tier, path=path)
    print(f"Set {ENV_LIMITS_TIER}={tier} in {path}")
    return tier
