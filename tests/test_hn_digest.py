import base64
import types
import uuid

import pytest

import mistralai.workflows as workflows
from mistralai.workflows.testing import create_test_worker

from worker.workflows import hn_digest
from worker.workflows.hn_digest import (
    HNDigestInput,
    HNDigestWorkflow,
    Story,
    apply_selection,
    build_animated_html,
    build_email_payload,
    build_selection_prompt,
    clamp_script,
    extract_json,
    filter_candidates,
    hn_source_url,
    kicker_for,
    markdown_to_html,
    parse_front_page,
)

SAMPLE_ALGOLIA = {
    "hits": [
        {
            "objectID": "111",
            "title": "OpenAI releases GPT-6",
            "url": "https://openai.com/blog/gpt-6",
            "author": "alice",
            "points": 812,
            "num_comments": 431,
        },
        {
            "objectID": "222",
            "title": "My thoughts on vim",
            "url": "https://jane.substack.com/p/vim",
            "author": "jane",
            "points": 50,
            "num_comments": 12,
        },
        {
            "objectID": "333",
            "title": "Mistral releases Pixtral 2 open vision model",
            "url": "https://mistral.ai/news/pixtral-2",
            "author": "bob",
            "points": 30,
            "num_comments": 5,
        },
        {
            "objectID": "444",
            "title": "Breaking: election results",
            "url": "https://twitter.com/news/status/1",
            "author": "x",
            "points": 900,
            "num_comments": 800,
        },
    ]
}


def test_parse_front_page():
    stories = parse_front_page(SAMPLE_ALGOLIA)
    assert [s.id for s in stories] == [111, 222, 333, 444]
    assert stories[0].domain == "openai.com"
    assert stories[2].url is None and stories[2].domain is None
    assert stories[2].hn_url == "https://news.ycombinator.com/item?id=333"


def test_filter_candidates_drops_social_and_personal_blogs():
    stories = parse_front_page(SAMPLE_ALGOLIA)
    kept = filter_candidates(stories)
    assert [s.id for s in kept] == [111, 333]


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"selected_ids": [1, 2]} hope that helps') == {"selected_ids": [1, 2]}


def test_apply_selection_orders_and_filters():
    candidates = parse_front_page(SAMPLE_ALGOLIA)
    candidates = filter_candidates(candidates)
    selected = apply_selection({"selected_ids": [333, 111, 999, "bogus"]}, candidates, 12)
    assert [s.id for s in selected] == [333, 111]


def test_apply_selection_falls_back_to_engagement_ranking():
    candidates = parse_front_page(SAMPLE_ALGOLIA)
    selected = apply_selection({}, candidates, 2)
    assert [s.id for s in selected] == [444, 111]


def test_selection_prompt_lists_all_candidates():
    candidates = filter_candidates(parse_front_page(SAMPLE_ALGOLIA))
    system, user = build_selection_prompt(candidates, 12)
    assert "id=111" in user and "id=333" in user
    assert "TECH INDUSTRY" in user
    assert "personal blogs" in user


def test_clamp_script_truncates_at_sentence():
    script = " ".join(f"word{i}." for i in range(500))
    clamped = clamp_script(script, max_words=50)
    assert len(clamped.split()) <= 50
    assert clamped.endswith(".")


def test_markdown_to_html_escapes_and_links():
    md = '1. **Big <news>** — see [source](https://example.com/a?b=1) and [HN](https://news.ycombinator.com/item?id=1)\n\n<script>alert(1)</script>'
    out = markdown_to_html(md)
    assert "<strong>Big &lt;news&gt;</strong>" in out
    assert '<a href="https://example.com/a?b=1">source</a>' in out
    assert "<script>" not in out


def test_build_email_payload_shape():
    audio_b64 = base64.b64encode(b"fake-mp3-bytes").decode()
    html_b64 = base64.b64encode(b"<html>animated</html>").decode()
    payload = build_email_payload(
        "me@example.com", "Subject", "# Hi", audio_b64, "digest.mp3", html_b64, "digest.html"
    )
    assert payload["to"] == ["me@example.com"]
    assert payload["subject"] == "Subject"
    assert len(payload["attachments"]) == 2
    assert payload["attachments"][0]["mimeType"] == "audio/mpeg"
    assert payload["attachments"][0]["filename"] == "digest.mp3"
    assert base64.b64decode(payload["attachments"][0]["content"], validate=True) == b"fake-mp3-bytes"
    assert payload["attachments"][1]["mimeType"] == "text/html"
    assert payload["attachments"][1]["filename"] == "digest.html"
    assert base64.b64decode(payload["attachments"][1]["content"], validate=True) == b"<html>animated</html>"

    payload_audio_only = build_email_payload("me@example.com", "S", "b", audio_b64, "a.mp3")
    assert len(payload_audio_only["attachments"]) == 1


def test_hn_source_url_mapping():
    assert "tags=front_page" in hn_source_url("front_page")
    assert "search_by_date" in hn_source_url("latest")
    assert "tags=story" in hn_source_url("best")
    with pytest.raises(ValueError):
        hn_source_url("nope")


def test_filter_candidates_drops_non_news_titles():
    stories = [
        Story(id=1, title="Ask HN: what laptop should I buy?", hn_url="https://news.ycombinator.com/item?id=1", points=50, num_comments=90),
        Story(id=2, title="Tell HN: I quit my job", hn_url="https://news.ycombinator.com/item?id=2", points=30, num_comments=40),
        Story(id=3, title="Show HN: I built a terminal in Rust", hn_url="https://news.ycombinator.com/item?id=3", points=120, num_comments=60),
        Story(id=4, title="Nvidia unveils new Blackwell Ultra chip", url="https://nvidia.com/news", domain="nvidia.com", hn_url="https://news.ycombinator.com/item?id=4", points=400, num_comments=150),
    ]
    kept = filter_candidates(stories)
    assert [s.id for s in kept] == [3, 4]


def test_kicker_for_categories():
    def story(title, domain="example.com"):
        return Story(id=1, title=title, hn_url="https://news.ycombinator.com/item?id=1", domain=domain)

    assert kicker_for(story("Gemini 5 LLM released"))[0] == "AI"
    assert kicker_for(story("New Firecracker VM escape exploit"))[0] == "SECURITY"
    assert kicker_for(story("Startup raises $500M round"))[0] == "BUSINESS"
    assert kicker_for(story("GitHub unveils new storage backend"))[0] == "DEV & INFRA"
    assert kicker_for(story("Tesla robotaxi delayed"))[0] == "BIG TECH"
    assert kicker_for(story("Tesla to wind down Cybertruck production"))[0] == "BIG TECH"
    assert kicker_for(story("New cyber-attack campaign spotted"))[0] == "SECURITY"
    assert kicker_for(story("RISC-V board ships"))[0] == "TECH"
    assert kicker_for(story("Anthropic publishes research", "anthropic.com"))[0] == "AI"


def test_build_animated_html_is_self_contained_and_escaped():
    audio_b64 = base64.b64encode(b"mp3bytes").decode()
    stories = [
        Story(id=1, title="<script>alert(1)</script> Gemini LLM news", hn_url="https://news.ycombinator.com/item?id=1", url="https://example.com/a", domain="example.com", points=10, num_comments=2),
        Story(id=2, title="Chips", hn_url="https://news.ycombinator.com/item?id=2", domain="vendor.io", points=5, num_comments=1),
    ]
    page = build_animated_html("HN Tech Digest - 2026-11-07", "# Digest", "Here is your digest.", stories, audio_b64, "2026-11-07")
    assert f"data:audio/mpeg;base64,{audio_b64}" in page
    assert page.count('class="card"') == 2
    assert 'id="card-0"' in page and 'id="card-1"' in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<script>alert" not in page
    assert 'id="narration"' in page and 'id="playbtn"' in page and 'id="pbar"' in page and 'id="clock"' in page
    assert 'src="http' not in page and "<link" not in page
    assert 'class="masthead"' in page and "TOP 2 TODAY" in page and "2026-11-07" in page
    assert 'class="kicker"' in page and ">AI<" in page
    assert "Here is your digest." in page


def _fake_client_with_chat(content: str):
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    response = types.SimpleNamespace(choices=[choice])

    class Chat:
        async def complete_async(self, **kwargs):
            return response

    class Client:
        chat = Chat()

    return Client()


async def test_llm_chat_json_parses_model_output(monkeypatch):
    monkeypatch.setattr(hn_digest, "_mistral_client", lambda **kwargs: _fake_client_with_chat('```json\n{"ok": true}\n```'))
    result = await hn_digest.llm_chat_json("sys", "user")
    assert result == {"ok": True}


async def test_tts_mp3_b64_validates_base64(monkeypatch):
    good = base64.b64encode(b"audio").decode()

    class Speech:
        async def complete_async(self, **kwargs):
            assert kwargs["response_format"] == "mp3"
            assert kwargs["voice_id"] == "voice_478"
            return types.SimpleNamespace(audio_data=good)

    class Client:
        audio = types.SimpleNamespace(speech=Speech())

    monkeypatch.setattr(hn_digest, "_mistral_client", lambda **kwargs: Client())
    assert await hn_digest.tts_mp3_b64("hello", "voice_478") == good

    class BadSpeech:
        async def complete_async(self, **kwargs):
            return types.SimpleNamespace(audio_data="!!!not-base64!!!")

    class BadClient:
        audio = types.SimpleNamespace(speech=BadSpeech())

    monkeypatch.setattr(hn_digest, "_mistral_client", lambda **kwargs: BadClient())
    with pytest.raises(Exception):
        await hn_digest.tts_mp3_b64("hello", "voice_478")


async def test_gmail_send_extracts_message_id():
    recorded = {}

    class FakeGmail:
        async def call_tool(self, tool_name, arguments=None):
            recorded["tool_name"] = tool_name
            recorded["arguments"] = arguments
            return {"id": "msg-123", "threadId": "t-1"}

    payload = build_email_payload("me@example.com", "s", "b", base64.b64encode(b"x").decode(), "a.mp3")
    message_id = await hn_digest.gmail_send(payload, FakeGmail())
    assert message_id == "msg-123"
    assert recorded["tool_name"] == "send_email"
    assert recorded["arguments"]["to"] == ["me@example.com"]


async def test_workflow_end_to_end_with_mocked_activities(temporal_env_with_converter):
    calls: dict = {}

    @workflows.activity(name="hn-fetch-front-page")
    async def fake_fetch(source: str = "front_page") -> list[Story]:
        calls["fetch"] = True
        calls["fetch_source"] = source
        return [
            Story(
                id=1,
                title="AI chip startup raises $500M",
                url="https://techcrunch.com/chip",
                domain="techcrunch.com",
                hn_url="https://news.ycombinator.com/item?id=1",
                points=500,
                num_comments=200,
            )
        ]

    @workflows.activity(name="hn-select-tech-stories")
    async def fake_select(stories: list[Story], count: int) -> list[Story]:
        calls["select"] = (len(stories), count)
        return list(stories)[:count]

    @workflows.activity(name="hn-build-digest")
    async def fake_build(stories: list[Story], date_str: str) -> dict:
        calls["build"] = (len(stories), date_str)
        return {
            "subject": f"HN Tech Digest - {date_str}",
            "markdown": "# Digest\n\n1. **AI chip startup raises $500M** — big round.",
            "script": "Here is your Hacker News tech digest.",
        }

    @workflows.activity(name="hn-render-audio-and-email")
    async def fake_render(to: str, digest: dict, stories: list[Story], voice_id: str, date_str: str) -> dict:
        calls["render"] = {"to": to, "voice_id": voice_id, "subject": digest["subject"], "stories": len(stories)}
        return {
            "sent_message_id": "msg-1",
            "audio_filename": f"hn-tech-digest-{date_str}.mp3",
            "audio_bytes": 12345,
            "html_filename": f"hn-tech-digest-{date_str}.html",
            "html_bytes": 6789,
        }

    env = temporal_env_with_converter
    async with create_test_worker(
        env,
        workflows=[HNDigestWorkflow],
        activities=[fake_fetch, fake_select, fake_build, fake_render],
    ):
        handle = await env.client.start_workflow(
            "hn-tech-digest",
            HNDigestInput(to="me@example.com", story_count=12, voice_id="voice_478"),
            id=f"test-{uuid.uuid4()}",
            task_queue="test-task-queue",
        )
        result = await handle.result()

    result = result if isinstance(result, hn_digest.DigestOutput) else hn_digest.DigestOutput(**result)
    assert result.sent_message_id == "msg-1"
    assert result.audio_filename.startswith("hn-tech-digest-")
    assert result.audio_bytes == 12345
    assert [s.id for s in result.stories] == [1]
    assert result.subject.startswith("HN Tech Digest -")
    assert calls["fetch"] is True
    assert calls["fetch_source"] == "latest"  # new default source
    assert calls["select"] == (1, 12)
    assert calls["render"]["to"] == "me@example.com"
    assert calls["render"]["voice_id"] == "voice_478"
    assert calls["render"]["stories"] == 1
    assert result.html_filename is not None and result.html_filename.endswith(".html")
    assert result.html_bytes == 6789
