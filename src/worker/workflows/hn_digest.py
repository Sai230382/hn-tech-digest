import base64
import html
import json
import re
from datetime import timedelta
from typing import Any

import pydantic

import mistralai.workflows as workflows
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

GMAIL_CONNECTOR_NAME = "gmail"
gmail_connector = connector(GMAIL_CONNECTOR_NAME)

CHAT_MODEL = "mistral-large-latest"
TTS_MODEL = "voxtral-mini-tts-3-0"
DEFAULT_VOICE_ID = "voice_478"
DEFAULT_RECIPIENT = "agmpbl10.saiu@iima.ac.in"
DEFAULT_STORY_COUNT = 12
FRONT_PAGE_SIZE = 30
MAX_SCRIPT_WORDS = 320

HN_FRONT_PAGE_URL = f"https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage={FRONT_PAGE_SIZE}"
HN_ITEM_URL = "https://news.ycombinator.com/item?id={object_id}"

NON_NEWS_DOMAINS = {
    "youtube.com",
    "youtu.be",
    "twitter.com",
    "x.com",
    "facebook.com",
    "instagram.com",
    "reddit.com",
    "tiktok.com",
}

PERSONAL_BLOG_DOMAINS = {
    "medium.com",
    "substack.com",
    "wordpress.com",
    "blogspot.com",
    "tumblr.com",
    "dev.to",
    "hashnode.com",
    "ghost.io",
}


class HNDigestInput(pydantic.BaseModel):
    to: str = DEFAULT_RECIPIENT
    story_count: int = DEFAULT_STORY_COUNT
    voice_id: str = DEFAULT_VOICE_ID


class Story(pydantic.BaseModel):
    id: int
    title: str
    url: str | None = None
    domain: str | None = None
    hn_url: str
    author: str | None = None
    points: int = 0
    num_comments: int = 0


class DigestOutput(pydantic.BaseModel):
    subject: str
    markdown: str
    script: str
    audio_filename: str
    audio_bytes: int
    html_filename: str | None = None
    html_bytes: int | None = None
    stories: list[Story]
    sent_message_id: str | None = None


def extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    decoder = json.JSONDecoder()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in LLM output: {text[:200]}")
    obj, _ = decoder.raw_decode(cleaned[start:])
    if not isinstance(obj, dict):
        raise ValueError("LLM output JSON is not an object")
    return obj


def _mistral_client():
    from mistralai.workflows.client import get_mistral_client

    return get_mistral_client()


def domain_of(url: str | None) -> str | None:
    if not url:
        return None
    from urllib.parse import urlparse

    host = urlparse(url).hostname or ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def parse_front_page(payload: dict[str, Any]) -> list[Story]:
    stories: list[Story] = []
    for hit in payload.get("hits", []):
        title = (hit.get("title") or "").strip()
        object_id = hit.get("objectID")
        if not title or object_id is None:
            continue
        url = hit.get("url")
        stories.append(
            Story(
                id=int(object_id),
                title=title,
                url=url,
                domain=domain_of(url),
                hn_url=HN_ITEM_URL.format(object_id=object_id),
                author=hit.get("author"),
                points=int(hit.get("points") or 0),
                num_comments=int(hit.get("num_comments") or 0),
            )
        )
    return stories


BLOCKED_DOMAINS = NON_NEWS_DOMAINS | PERSONAL_BLOG_DOMAINS


def filter_candidates(stories: list[Story]) -> list[Story]:
    kept: list[Story] = []
    for story in stories:
        domain = story.domain or ""
        if any(domain == d or domain.endswith("." + d) for d in BLOCKED_DOMAINS):
            continue
        kept.append(story)
    return kept


def build_selection_prompt(stories: list[Story], count: int) -> tuple[str, str]:
    listing = "\n".join(
        f'- id={s.id} | {s.points} pts, {s.num_comments} comments | domain={s.domain or "news.ycombinator.com"} | {s.title}'
        for s in stories
    )
    system = (
        "You are an editor for a tech-industry news digest. You receive Hacker News front-page "
        "stories and must select the ones that are genuinely tech-industry news."
    )
    user = f"""Select up to {count} stories from this Hacker News front-page list that are TECH INDUSTRY news:
AI/ML products and research, big-tech company moves (Apple, Google, Microsoft, Amazon, Meta, Nvidia, Tesla, etc.),
startups and funding rounds, developer tools and infrastructure, hardware and chips, open-source projects with
industry impact, cybersecurity, cloud, mobile platforms, and similar.

EXCLUDE: personal blogs and individual essays/opinion pieces, non-tech topics (politics, general science,
sports, lifestyle, memes), pure hype or price speculation, and "Ask HN"/"Tell HN" threads unless they are
clearly about a substantive tech-industry topic. Stories from personal-blog platforms are excluded unless
they are a major industry announcement.

Order the selected ids by importance for a tech professional.

Stories:
{listing}

Reply with JSON only: {{"selected_ids": [<id>, <id>, ...]}} using the numeric ids above."""
    return system, user


def apply_selection(raw: dict[str, Any], candidates: list[Story], count: int) -> list[Story]:
    by_id = {s.id: s for s in candidates}
    selected: list[Story] = []
    raw_ids = raw.get("selected_ids")
    if isinstance(raw_ids, list):
        for item in raw_ids:
            try:
                story = by_id.get(int(item))
            except (TypeError, ValueError):
                continue
            if story is not None and story not in selected:
                selected.append(story)
            if len(selected) >= count:
                break
    if not selected:
        ranked = sorted(candidates, key=lambda s: (s.points + 2 * s.num_comments), reverse=True)
        selected = ranked[:count]
    return selected[:count]


def build_digest_prompt(stories: list[Story], date_str: str) -> tuple[str, str]:
    listing = "\n".join(
        f'{i + 1}. {s.title} (domain: {s.domain or "news.ycombinator.com"}, {s.points} pts, {s.num_comments} comments, source: {s.url or s.hn_url}, HN discussion: {s.hn_url})'
        for i, s in enumerate(stories)
    )
    system = (
        "You write a concise daily tech-news digest for a busy tech professional, plus a short "
        "podcast-style audio script that reads the headlines aloud."
    )
    user = f"""Today is {date_str}. Below are the selected Hacker News tech-industry stories.

Produce a JSON object with exactly these keys:
- "subject": email subject line, e.g. "HN Tech Digest - {date_str}"
- "markdown": the email digest. Numbered list, one entry per story: bold headline, one or two
  sentences of summary, then the source link and the HN discussion link. End with a one-line footer
  "Audio briefing attached (MP3, about 1-2 minutes)."
- "script": a spoken podcast-style script for the audio briefing. Start with a short intro like
  "Here is your Hacker News tech digest for {date_str}." Then read each headline with a single
  sentence of context. Close with "That's all for today. See you tomorrow." Plain text only: no
  markdown, no URLs, no bullet symbols. Target {MAX_SCRIPT_WORDS - 60}-{MAX_SCRIPT_WORDS} words total
  (about 1 to 2 minutes at a normal speaking pace). Do not exceed {MAX_SCRIPT_WORDS} words.

Stories:
{listing}

Reply with JSON only."""
    return system, user


def clamp_script(script: str, max_words: int = MAX_SCRIPT_WORDS) -> str:
    words = script.split()
    if len(words) <= max_words:
        return script
    truncated = " ".join(words[:max_words])
    last_period = truncated.rfind(".")
    if last_period > 0:
        truncated = truncated[: last_period + 1]
    return truncated


def markdown_to_html(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("<br>")
            continue
        text = html.escape(stripped)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', text)
        if stripped.startswith("#"):
            out.append(f"<h2>{text.lstrip('#').strip()}</h2>")
        else:
            out.append(f"<p>{text}</p>")
    return "<html><body>" + "".join(out) + "</body></html>"


_ANIM_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; font-family: 'Segoe UI', Arial, Helvetica, sans-serif; background: radial-gradient(1200px 600px at 20% -10%, #1b2a4a 0%, #0b0f1a 55%, #05070d 100%); color: #e8ecf4; min-height: 100vh; }
#pwrap { position: fixed; top: 0; left: 0; right: 0; height: 4px; background: rgba(255,255,255,.08); z-index: 10; }
#pbar { height: 100%; width: 0; background: linear-gradient(90deg, #4da3ff, #7c5cff); transition: width .2s linear; }
header { text-align: center; padding: 40px 16px 12px; }
.brand { font-size: 30px; letter-spacing: 5px; font-weight: 800; }
.dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: #ff3b30; margin-right: 10px; animation: pulse 1.2s infinite; }
@keyframes pulse { 50% { opacity: .25; } }
.date { color: #9aa7bd; letter-spacing: 2px; font-size: 13px; margin-left: 12px; }
.sub { color: #c7d2e4; margin-top: 8px; font-size: 15px; }
main { padding: 10px 16px 140px; }
.card { display: flex; gap: 14px; background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09); border-radius: 14px; padding: 16px 18px; margin: 0 auto 14px; max-width: 880px; opacity: 0; transform: translateY(26px); animation: cardIn .7s cubic-bezier(.2,.7,.2,1) forwards; }
@keyframes cardIn { to { opacity: 1; transform: none; } }
.card.active { border-color: #4da3ff; box-shadow: 0 0 0 1px #4da3ff, 0 10px 34px rgba(77,163,255,.28); background: rgba(77,163,255,.09); }
.num { font-size: 28px; font-weight: 800; color: #4da3ff; min-width: 46px; }
.ctitle { font-size: 20px; font-weight: 700; line-height: 1.3; }
.cmeta { color: #9aa7bd; font-size: 13px; margin-top: 6px; }
.cmeta a { color: #7fb4ff; }
.controls { position: sticky; bottom: 58px; display: flex; justify-content: center; gap: 14px; align-items: center; padding: 12px; z-index: 5; }
#playbtn { font-size: 16px; padding: 10px 24px; border-radius: 999px; border: none; background: #4da3ff; color: #04122b; font-weight: 700; cursor: pointer; }
#time { color: #9aa7bd; font-size: 13px; font-variant-numeric: tabular-nums; }
details.digest { max-width: 880px; margin: 20px auto; padding: 0 4px; color: #c7d2e4; }
details.digest summary { cursor: pointer; color: #7fb4ff; font-weight: 600; }
.digest-body { background: rgba(255,255,255,.04); border-radius: 12px; padding: 14px 18px; margin-top: 10px; }
.digest-body p { line-height: 1.55; }
.script { margin-top: 14px; color: #9aa7bd; font-size: 14px; line-height: 1.6; }
.ticker { position: fixed; bottom: 0; left: 0; right: 0; overflow: hidden; background: rgba(0,0,0,.6); border-top: 1px solid rgba(255,255,255,.1); padding: 10px 0; z-index: 9; }
.ticker-inner { display: inline-block; white-space: nowrap; animation: marquee 40s linear infinite; color: #9aa7bd; font-size: 13px; letter-spacing: 1px; }
@keyframes marquee { from { transform: translateX(0); } to { transform: translateX(-50%); } }
"""

_ANIM_JS = """
const audio = document.getElementById('narration');
const cards = Array.from(document.querySelectorAll('.card'));
const bar = document.getElementById('pbar');
const btn = document.getElementById('playbtn');
const timeEl = document.getElementById('time');
let lastActive = -1;
function fmt(t) { if (!isFinite(t)) return '0:00'; const m = Math.floor(t / 60), s = Math.floor(t % 60); return m + ':' + String(s).padStart(2, '0'); }
btn.addEventListener('click', () => { if (audio.paused) { audio.play(); } else { audio.pause(); } });
audio.addEventListener('play', () => { btn.textContent = '\\u275A\\u275A Pause'; });
audio.addEventListener('pause', () => { btn.textContent = '\\u25B6 Play narration'; });
audio.addEventListener('loadedmetadata', () => {
  const d = audio.duration || 0;
  cards.forEach((c, i) => {
    c.dataset.start = (i * d / cards.length).toString();
    c.dataset.end = ((i + 1) * d / cards.length).toString();
  });
});
audio.addEventListener('timeupdate', () => {
  const t = audio.currentTime, d = audio.duration || 0;
  bar.style.width = (d ? (t / d) * 100 : 0) + '%';
  timeEl.textContent = fmt(t) + ' / ' + fmt(d);
  let active = -1;
  cards.forEach((c, i) => {
    const s = parseFloat(c.dataset.start || '0'), e = parseFloat(c.dataset.end || '0');
    if (t >= s && t < e) { active = i; }
  });
  cards.forEach((c, i) => c.classList.toggle('active', i === active));
  if (active !== lastActive && active >= 0) { cards[active].scrollIntoView({ block: 'nearest', behavior: 'smooth' }); lastActive = active; }
});
audio.addEventListener('ended', () => { cards.forEach(c => c.classList.remove('active')); lastActive = -1; });
"""


def build_animated_html(
    subject: str,
    markdown: str,
    script: str,
    stories: list[Story],
    audio_b64: str,
    date_str: str,
) -> str:
    cards_html = []
    for i, s in enumerate(stories):
        title = html.escape(s.title)
        dom = html.escape(s.domain or "news.ycombinator.com")
        hn = html.escape(s.hn_url, quote=True)
        src_link = f' &middot; <a href="{html.escape(s.url, quote=True)}">source</a>' if s.url else ""
        cards_html.append(
            f'<div class="card" id="card-{i}" style="animation-delay:{i * 5.0:.1f}s">'
            f'<div class="num">{i + 1}</div>'
            f'<div class="cbody"><div class="ctitle">{title}</div>'
            f'<div class="cmeta">{dom} &middot; {s.points} pts &middot; {s.num_comments} comments &middot; '
            f'<a href="{hn}">HN discussion</a>{src_link}</div></div></div>'
        )
    ticker = " &nbsp;&#8226;&nbsp; ".join(html.escape(s.domain or "news.ycombinator.com") for s in stories)
    ticker = ticker + " &nbsp;&#8226;&nbsp; " + ticker
    digest_body = markdown_to_html(markdown).removeprefix("<html><body>").removesuffix("</body></html>")
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(subject)}</title><style>{_ANIM_CSS}</style></head><body>"
        "<div id=\"pwrap\"><div id=\"pbar\"></div></div>"
        f"<header><div class=\"brand\"><span class=\"dot\"></span>HN TECH DIGEST"
        f"<span class=\"date\">{html.escape(date_str)}</span></div>"
        f"<div class=\"sub\">{html.escape(subject)}</div></header>"
        "<main>" + "".join(cards_html) + "</main>"
        "<div class=\"controls\"><button id=\"playbtn\">&#9654; Play narration</button>"
        "<span id=\"time\">0:00 / 0:00</span></div>"
        f"<audio id=\"narration\" preload=\"auto\" src=\"data:audio/mpeg;base64,{audio_b64}\"></audio>"
        "<details class=\"digest\"><summary>Full digest text &amp; transcript</summary>"
        f"<div class=\"digest-body\">{digest_body}</div>"
        f"<div class=\"script\"><strong>Narration transcript</strong><br>{html.escape(script)}</div></details>"
        f"<footer class=\"ticker\"><div class=\"ticker-inner\">{ticker}</div></footer>"
        f"<script>{_ANIM_JS}</script></body></html>"
    )


def build_email_payload(
    to: str,
    subject: str,
    markdown: str,
    audio_b64: str,
    audio_filename: str,
    html_b64: str | None = None,
    html_filename: str | None = None,
) -> dict[str, Any]:
    attachments: list[dict[str, Any]] = [
        {"content": audio_b64, "filename": audio_filename, "mimeType": "audio/mpeg"}
    ]
    if html_b64 is not None and html_filename is not None:
        attachments.append({"content": html_b64, "filename": html_filename, "mimeType": "text/html"})
    return {
        "to": [to],
        "subject": subject,
        "body": markdown,
        "htmlBody": markdown_to_html(markdown),
        "attachments": attachments,
    }


async def hn_get_json(url: str) -> Any:
    import httpx

    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "hn-tech-digest/0.1"}) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


async def llm_chat_json(system: str, user: str) -> dict[str, Any]:
    client = _mistral_client()
    response = await client.chat.complete_async(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return extract_json(content)


async def tts_mp3_b64(script: str, voice_id: str) -> str:
    client = _mistral_client()
    response = await client.audio.speech.complete_async(
        model=TTS_MODEL,
        input=script,
        voice_id=voice_id,
        response_format="mp3",
    )
    audio_b64 = response.audio_data
    base64.b64decode(audio_b64, validate=True)
    return audio_b64


async def gmail_send(payload: dict[str, Any], gmail: ToolCallClient) -> str:
    result = await gmail.call_tool(tool_name="send_email", arguments=payload)
    if isinstance(result, dict):
        return str(result.get("id"))
    return str(result)


@workflows.activity(name="hn-fetch-front-page", start_to_close_timeout=timedelta(seconds=60))
async def fetch_front_page() -> list[Story]:
    payload = await hn_get_json(HN_FRONT_PAGE_URL)
    return parse_front_page(payload)


@workflows.activity(name="hn-select-tech-stories", start_to_close_timeout=timedelta(seconds=180))
async def select_tech_stories(stories: list[Story], count: int) -> list[Story]:
    candidates = filter_candidates(stories)
    system, user = build_selection_prompt(candidates, count)
    raw = await llm_chat_json(system, user)
    return apply_selection(raw, candidates, count)


@workflows.activity(name="hn-build-digest", start_to_close_timeout=timedelta(seconds=180))
async def build_digest(stories: list[Story], date_str: str) -> dict[str, Any]:
    system, user = build_digest_prompt(stories, date_str)
    digest = await llm_chat_json(system, user)
    digest["script"] = clamp_script(str(digest.get("script", "")))
    return digest


@workflows.activity(name="hn-render-audio-and-email", start_to_close_timeout=timedelta(minutes=6))
async def render_audio_and_email(
    to: str,
    digest: dict[str, Any],
    stories: list[Story],
    voice_id: str,
    date_str: str,
    gmail: ToolCallClient = Depends(gmail_connector),
) -> dict[str, Any]:
    audio_b64 = await tts_mp3_b64(digest["script"], voice_id)
    audio_bytes = len(base64.b64decode(audio_b64))
    audio_filename = f"hn-tech-digest-{date_str}.mp3"
    subject = str(digest.get("subject", f"HN Tech Digest - {date_str}"))
    animated = build_animated_html(
        subject, str(digest.get("markdown", "")), str(digest.get("script", "")), stories, audio_b64, date_str
    )
    html_filename = f"hn-tech-digest-{date_str}.html"
    html_b64 = base64.b64encode(animated.encode("utf-8")).decode("ascii")
    payload = build_email_payload(
        to, subject, str(digest["markdown"]), audio_b64, audio_filename, html_b64, html_filename
    )
    message_id = await gmail_send(payload, gmail)
    return {
        "sent_message_id": message_id,
        "audio_filename": audio_filename,
        "audio_bytes": audio_bytes,
        "html_filename": html_filename,
        "html_bytes": len(animated.encode("utf-8")),
    }


@workflows.workflow.define(
    name="hn-tech-digest",
    workflow_display_name="HN Tech Digest",
    workflow_description=(
        "One-shot digest: reads the Hacker News front page, selects the top tech-industry stories, "
        "summarizes them, renders a 1-2 minute Voxtral TTS audio briefing plus an animated HTML "
        "news-style video page synced to the narration, and emails the digest with the MP3 and HTML "
        "attached to your Gmail inbox. Runs on behalf of the triggering user."
    ),
    on_behalf_of=True,
)
@uses_connectors(gmail_connector)
class HNDigestWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: HNDigestInput) -> DigestOutput:
        date_str = workflows.workflow.now().strftime("%Y-%m-%d")
        stories = await fetch_front_page()
        selected = await select_tech_stories(stories, input.story_count)
        digest = await build_digest(selected, date_str)
        sent = await render_audio_and_email(input.to, digest, selected, input.voice_id, date_str)
        return DigestOutput(
            subject=str(digest.get("subject", f"HN Tech Digest - {date_str}")),
            markdown=str(digest.get("markdown", "")),
            script=str(digest.get("script", "")),
            audio_filename=sent["audio_filename"],
            audio_bytes=sent["audio_bytes"],
            html_filename=sent["html_filename"],
            html_bytes=sent["html_bytes"],
            stories=selected,
            sent_message_id=sent["sent_message_id"],
        )
