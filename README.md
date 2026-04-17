# capsule-litellm

[![AI Agents](https://img.shields.io/badge/AI%20Agents-AGENTS.md-blueviolet.svg)](./AGENTS.md)

[Capsule Protocol](https://github.com/quantumpipes/capsule) integration for [LiteLLM](https://github.com/BerriAI/litellm).

Every LLM call through LiteLLM automatically produces a sealed, hash-chained Capsule. Zero-config audit trail.

> **AI coding agents:** start with [AGENTS.md](./AGENTS.md). It contains commands, the tiny integration surface, and the "audit must never break the user's LLM call" invariant.

## Explore the fit with your AI coding agent

Paste this prompt into Claude Code, Cursor, Codex, or any other agent:

```text
Read the capsule-litellm README and AGENTS.md at https://github.com/quantumpipes/capsule-litellm.
Then find every place my codebase calls litellm.completion or litellm.acompletion.
Show me the two-line change per entry point to register CapsuleLogger, flag any
callers that already set litellm.callbacks (conflict risk), and propose where the
resulting Capsule chain should be stored.
```

---

## Install

```bash
pip install capsule-litellm
```

## Quick Start

```python
import litellm
from capsule_litellm import CapsuleLogger

# Register the callback
litellm.callbacks = [CapsuleLogger()]

# Use LiteLLM normally -- Capsules are created automatically
response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
```

Every call now produces a sealed Capsule with:

| Section | Content |
|---|---|
| **Trigger** | User's message, timestamp, source |
| **Context** | Agent ID, model, call type |
| **Reasoning** | Model selection, SHA3-256 prompt hash |
| **Authority** | Autonomous (default) |
| **Execution** | LiteLLM call, duration, token counts |
| **Outcome** | Response text, success/failure, metrics |

## Configuration

```python
from qp_capsule import Capsules, CapsuleType
from capsule_litellm import CapsuleLogger

# Custom storage, agent identity, and domain
logger = CapsuleLogger(
    capsules=Capsules("postgresql://..."),
    agent_id="trading-bot",
    domain="finance",
    capsule_type=CapsuleType.AGENT,
    swallow_errors=True,  # default: don't interrupt LLM calls on capsule failure
)

litellm.callbacks = [logger]
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `capsules` | `Capsules()` | Storage, chain, and seal instance (SQLite default) |
| `agent_id` | `"litellm"` | Identity in the capsule's context section |
| `domain` | `"chat"` | Business domain for filtering |
| `capsule_type` | `CapsuleType.CHAT` | Capsule type for each record |
| `swallow_errors` | `True` | Log capsule errors instead of raising |

## What Gets Captured

### Prompt Hash (not the prompt)

The full prompt is hashed with SHA3-256 and stored as `reasoning.prompt_hash`. This enables prompt auditing without storing sensitive context in the audit trail.

### Token Metrics

Token counts from the LLM response are stored in both `execution.resources_used` and `outcome.metrics`:

```python
{
    "tokens_in": 150,
    "tokens_out": 42,
    "latency_ms": 1200,
}
```

### Error Tracking

Failed LLM calls are recorded with `outcome.status = "failure"` and the error message in `outcome.error`.

## Async Support

Both sync and async LiteLLM calls are captured:

```python
# Sync
response = litellm.completion(model="gpt-4o", messages=messages)

# Async
response = await litellm.acompletion(model="gpt-4o", messages=messages)
```

## Verification

Capsules are cryptographically sealed and hash-chained. Verify with the CPS CLI:

```bash
capsule verify chain.json --full
```

## License

Apache-2.0. See the [Capsule Protocol](https://github.com/quantumpipes/capsule) for specification and patent grant.
