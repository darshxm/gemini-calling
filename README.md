# gemini-calling

A Python library for calling the Gemini REST API with zero external dependencies and simple .env management for one or many API keys.

Highlights:
- Minimal: uses Python standard library (`urllib`) — no installs needed.
- .env helpers: create and manage a `.env` file; support multiple keys.
- Round-robin key selection when multiple API keys are configured.
- Focused scope: text generation (`generateContent` endpoint) for now.

## Installation

- Pip via HTTPS:
  - `pip install "git+https://github.com/darshxm/gemini-calling.git@main"`
- Pin to a tag/commit:
  - `pip install "git+https://github.com/darshxm/gemini-calling.git@v0.1.0"`
  - `pip install "git+https://github.com/darshxm/gemini-calling.git@<commit-sha>"`

Notes:

- Import name is `gemini_calling` (underscore): `from gemini_calling import gemini_request`.

## Quick start

```
from gemini_calling import gemini_request, create_env

# 1) (Optional) Create a .env file with placeholders or your keys
create_env(path=".env")  # will not overwrite by default

# 2) One-shot request that auto-loads keys from env/.env
res = gemini_request(prompt="How does AI work?", model="gemini-2.5-flash")
print(res.text)
```

## How it works

- What it is: a tiny helper that asks Gemini questions and returns answers.
- Your keys: put one or many in `.env` or pass them to `GeminiClient(api_keys=[...])`. The second option is not recommended, but handy for quick tests. Please never commit your API keys.
- Tiers: set `limits_tier="free"` or `"tier1"` (or `GEMINI_LIMITS_TIER`) so requests stay under Google’s limits. Support to be added for tier 2 and 3.
- Two modes per call (pick one):
  - Structured output (JSON/enums) via `response_mime_type` + `response_schema`/`response_enum`.
  - Function calling via `tools`, `function_calling_mode`, etc. Do not mix with structured output.
  - Coming soon: Grounding with google search, maps, url context capabilities, and code execution.
- Safety: automatic retries/backoff, key rotation, key disable on auth errors, and per-key + global rate limiting (RPM/RPD/TPM).
- Extras: system instructions, temperature/topP/topK/stop, thinking budget, or full `contents` control.

Quickstart:

```
from gemini_calling import gemini_request

res = gemini_request(prompt="How does AI work?", model="gemini-2.5-flash")
print(res.text)
```

Compatibility:
- Python >= 3.8
- No external dependencies (uses `urllib`)

## API Keys

You can provide keys in several ways:

- Environment variable `GEMINI_API_KEY` (single key)
- Environment variable `GEMINI_API_KEYS` (comma or newline separated for multiple keys)
- `.env` file with either `GEMINI_API_KEY=...` or `GEMINI_API_KEYS=key1,key2,...`
- Named keys (recommended): add entries like `GEMINI_API_KEY_MY_FREE_KEY=...` or `GEMINI_API_KEY_MY_PAID_KEY=...`

You can also set a tier to control client-side rate limits:

- `GEMINI_LIMITS_TIER=free` (default) or `tier1` (paid Tier 1)

Helpers:

```
from gemini_calling import create_env, load_keys, set_keys

# Write a new .env with two keys (won't overwrite unless overwrite=True)
create_env(["key-1", "key-2"], path=".env", overwrite=False)

# Update/replace keys in .env
set_keys(["key-1", "key-2", "key-3"], path=".env")

# Load keys from environment or .env
print(load_keys(path=".env"))
```

Named keys usage:

```
# .env
# GEMINI_API_KEY_MY_FREE_KEY=sk-...
# GEMINI_API_KEY_MY_PAID_KEY=sk-...

from gemini_calling import gemini_request

# Select a named key by its identifier (case-insensitive match is allowed)
res = gemini_request(
    prompt="Hello!",
    model="gemini-2.5-flash",
    key_name="MY_PAID_KEY",   # or "my_paid_key"
)
print(res.text)
```

## Advanced usage

```
from gemini_calling import gemini_request

# System instruction and parameters
res = gemini_request(
    prompt="Hello there",
    model="gemini-2.5-flash",
    system_instruction="You are a cat. Your name is Neko.",
    temperature=0.1,
)
print(res.text)

# Disable thinking (for 2.5‑Flash)
res = gemini_request(
    prompt="How does AI work?",
    model="gemini-2.5-flash",
    thinking_budget=0,
)
print(res.text)
```

## Structured output

Generate JSON using a response schema:

```
from gemini_calling import gemini_request

schema = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "recipeName": {"type": "STRING"},
            "ingredients": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "propertyOrdering": ["recipeName", "ingredients"],
    },
}

res = gemini_request(
    prompt="List a few popular cookie recipes, and include the amounts of ingredients.",
    model="gemini-2.5-flash",
    response_mime_type="application/json",
    response_schema=schema,
    estimated_output_tokens=400,
)

print(res.text)        # JSON string
print(res.parse_json())  # Parsed JSON (dict/list) or None
```

Generate enum values (single selection):

```
res = gemini_request(
    prompt="What type of instrument is an oboe?",
    model="gemini-2.5-flash",
    response_enum=["Percussion", "String", "Woodwind", "Brass", "Keyboard"],
)
print(res.text)
```

Pass JSON Schema directly (Gemini 2.5 only; preview):

```
example_json_schema = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "username": {"type": "string"},
    "age": {"type": "integer", "minimum": 0, "maximum": 120}
  },
  "required": ["username"]
}

res = gemini_request(
    prompt="Please give a random example following this schema",
    model="gemini-2.5-flash",
    response_mime_type="application/json",
    response_json_schema=example_json_schema,
)
print(res.parse_json())
```

## Function calling

Provide function declarations and receive function call suggestions:

```
from gemini_calling import gemini_request

schedule_meeting = {
    "name": "schedule_meeting",
    "description": "Schedules a meeting with specified attendees at a given time and date.",
    "parameters": {
        "type": "object",
        "properties": {
            "attendees": {"type": "array", "items": {"type": "string"}},
            "date": {"type": "string"},
            "time": {"type": "string"},
            "topic": {"type": "string"},
        },
        "required": ["attendees", "date", "time", "topic"],
    },
}

tools = [{"functionDeclarations": [schedule_meeting]}]

res = gemini_request(
    prompt=(
        "Schedule a meeting with Bob and Alice for 03/27/2025 at 10:00 AM "
        "about the Q3 planning."
    ),
    model="gemini-2.5-flash",
    tools=tools,
    function_calling_mode="AUTO",  # or "ANY" or "NONE"
)

if res.function_calls:
    call = res.function_calls[0]
    print("Function to call:", call["name"])  # e.g., schedule_meeting
    print("Args:", call["args"])            # dict
else:
    print(res.text)
```

Multi-turn with a function response:

```
# 1) First turn with tools
first = gemini_request(
    prompt="Turn the lights down to a romantic level",
    model="gemini-2.5-flash",
    tools=[{"functionDeclarations": [{
        "name": "set_light_values",
        "description": "Sets the brightness and color temperature of a light.",
        "parameters": {
            "type": "object",
            "properties": {
                "brightness": {"type": "integer"},
                "color_temp": {"type": "string", "enum": ["daylight","cool","warm"]},
            },
            "required": ["brightness","color_temp"],
        },
    }]}],
)

if first.function_calls:
    fc = first.function_calls[0]
    # Execute your own function here and get a result dict
    tool_result = {"brightness": 25, "colorTemperature": "warm"}

    # 2) Build contents for follow-up including previous model content and function response
    contents = []
    # Append the model's previous content
    contents.append(first.raw["candidates"][0]["content"])  # model turn
    # Append the function response as a user part
    contents.append({
        "role": "user",
        "parts": [{"functionResponse": {"name": fc["name"], "response": {"result": tool_result}}}],
    })

    final = gemini_request(
        contents=contents,
        model="gemini-2.5-flash",
        tools=[{"functionDeclarations": [{"name": "set_light_values", "parameters": {"type": "object"}}]}],
    )
    print(final.text)
else:
    print(first.text)
```

Notes:
- Do not combine function calling and structured output in the same request; this client enforces that.
- For complex flows, pass full `contents` for exact control of roles and parts.

## Retries, backoff, and key health

- The client retries on transient errors by default (HTTP 429/5xx, network errors) with exponential backoff.
- It rotates across multiple keys on retries when `rotate_keys_on_retry=True`.
- For HTTP 429, if the response includes a `Retry-After` header, the client respects it (when `respect_retry_after=True`).
- On HTTP 401/403 the key is marked disabled and not used again in this process.
- The client tracks simple per-key health: cooldowns, failures, disabled flag.

Inspect key health at runtime:

```
for h in client.key_health():
    print(h)
```

## Client-side rate limiting

Light, best‑effort limiter for common models (free and tier1 presets). It throttles per key and globally per model across:

- RPM: requests per minute (rolling 60s window)
- RPD: requests per day (rolling 24h window)
- TPM: tokens per minute using a token bucket (estimation)

Configure with `GEMINI_LIMITS_TIER=free|tier1` or pass `limits_tier`. Token‑per‑minute (TPM) enforcement is optional and OFF by default; enable it with `enforce_tpm=True`:

```
from gemini_calling import gemini_request, Limits

res = gemini_request(
    prompt="...",
    model="gemini-2.5-flash",
    limits_tier="tier1",  # or set GEMINI_LIMITS_TIER in .env
    # Optional: override/extend limits per model prefix
    model_limits={
        # "gemini-2.5-flash": Limits(rpm=60, tpm=1_000_000, rpd=10_000),
    },
    # Enforce TPM only if you want token buckets
    enforce_tpm=True,
    # Optional: better token planning for TPM when enabled
    estimated_output_tokens=300,
)
```

Notes:
- Limits match by longest model prefix (case‑insensitive). Unknown models aren’t throttled unless configured.
- TPM enforcement is off by default; when enabled, token planning uses a simple heuristic (~4 chars/token) unless you provide a custom estimator.


## Streaming

Not implemented yet. Please contribute with a PR, I am but one man.

## Disclaimer

This library is a small convenience wrapper and is not an official Google SDK. You are responsible for managing your API keys and usage. I am serious about the API key management, please DO NOT COMMIT.

```
Environment variables recognized:
- GEMINI_API_KEY
- GEMINI_API_KEYS
- GEMINI_LIMITS_TIER
```

## License

One of the open ones, I won't sue you.
