# hn-tech-digest

A [Mistral Workflow](https://docs.mistral.ai/studio/workflows/overview) that produces a one-shot
Hacker News tech digest:

1. Fetches the Hacker News front page (`hn.algolia.com/api/v1/search?tags=front_page`).
2. Filters out non-tech and personal-blog stories and picks the top N tech-industry headlines
   (LLM selection with deterministic fallbacks).
3. Writes a markdown digest and a 1-2 minute podcast-style script (Mistral chat model).
4. Renders the script to MP3 with Voxtral TTS (`voxtral-mini-tts-3-0`).
5. Sends the digest email with the MP3 attached to your Gmail inbox via the `gmail` connector,
   running on behalf of the triggering user (`on_behalf_of=True`).

It is deployed as a **hardened managed deployment** on Mistral Cloud and has **no schedule**: you
trigger a run manually (one-shot) from the Mistral Console ("Start workflow") or with:

```bash
curl -X POST https://api.mistral.ai/v1/workflows/hn-tech-digest/execute \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"to": "you@gmail.com", "story_count": 12, "voice_id": "voice_478"}}'
```

All input fields are optional; defaults: `to` = your connected Gmail address,
`story_count` = 12, `voice_id` = `voice_478` (Sheila, en-US).

## Layout

- `src/worker/workflows/hn_digest.py` — the `hn-tech-digest` workflow and its activities
- `src/worker/entrypoints/worker.py` — worker entrypoint (discovers workflows, runs the worker)
- `Dockerfile` — image built by the managed deployment; `CMD` starts the worker
- `tests/` — unit tests for the pure helpers plus a Temporal time-skipping test of the workflow
  with all external effects mocked

## Develop

```bash
uv sync --group dev
uv run pytest
```

## Deploy

Push to GitHub, then create a managed deployment from this repository (Dockerfile path
`Dockerfile`, build directory `.`), or via the Mistral MCP `deploy_workflow_deployment` tool with
`hardened=true` (required for on-behalf-of workflows). The Mistral GitHub App
(github.com/apps/mistralai) must be installed on this repository so Mistral Cloud can clone it.
