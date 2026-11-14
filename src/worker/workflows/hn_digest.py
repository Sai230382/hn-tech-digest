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

HN_SOURCES = {
    "front_page": f"https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage={FRONT_PAGE_SIZE}",
    "latest": f"https://hn.algolia.com/api/v1/search_by_date?tags=story&hitsPerPage={FRONT_PAGE_SIZE}",
    "best": f"https://hn.algolia.com/api/v1/search?tags=story&hitsPerPage={FRONT_PAGE_SIZE}",
}
HN_ITEM_URL = "https://news.ycombinator.com/item?id={object_id}"


def hn_source_url(source: str) -> str:
    try:
        return HN_SOURCES[source]
    except KeyError:
        raise ValueError(f"unknown HN source {source!r}; choose one of {sorted(HN_SOURCES)}") from None

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
    source: str = "latest"  # one of: "latest" (newest stories), "front_page", "best"


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


NON_NEWS_TITLE_PREFIXES = ("Ask HN:", "Tell HN:", "Poll:")


def filter_candidates(stories: list[Story]) -> list[Story]:
    kept: list[Story] = []
    for story in stories:
        domain = story.domain or ""
        if any(domain == d or domain.endswith("." + d) for d in BLOCKED_DOMAINS):
            continue
        if story.title.startswith(NON_NEWS_TITLE_PREFIXES):
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
:root { color-scheme: dark; --ink:#0d0d0f; --paper:#f5f1e8; --red:#ff3b30; --yellow:#ffd23f; --blue:#4da3ff; --muted:#9aa0ad; }
* { box-sizing: border-box; }
body { margin: 0; background-color: var(--ink); color: var(--paper); font-family: 'Segoe UI', Arial, Helvetica, sans-serif; background-image: radial-gradient(rgba(255,255,255,.045) 1px, transparent 1px); background-size: 22px 22px; min-height: 100vh; }
#pwrap { position: fixed; top: 0; left: 0; right: 0; height: 5px; background: rgba(255,255,255,.08); z-index: 20; }
#pbar { height: 100%; width: 0; background: var(--red); transition: width .2s linear; }
header { max-width: 920px; margin: 0 auto; padding: 26px 20px 6px; position: relative; }
.topbar { display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid var(--red); padding-bottom: 10px; }
.logo { font-weight: 900; letter-spacing: 2px; font-size: 20px; }
.logo em { color: var(--red); font-style: normal; }
.clock { color: var(--muted); font-size: 13px; letter-spacing: 1px; font-variant-numeric: tabular-nums; }
.masthead { font-family: Georgia, 'Times New Roman', serif; font-size: clamp(46px, 8vw, 86px); line-height: .95; letter-spacing: -2px; margin: 20px 0 8px; }
.masthead .accent { color: var(--red); }
.deck { color: var(--muted); font-size: 15px; }
.sticker { position: absolute; right: 10px; top: 92px; background: var(--yellow); color: #141414; font-weight: 800; font-size: 12px; letter-spacing: 1px; padding: 8px 12px; border-radius: 6px; transform: rotate(4deg); box-shadow: 3px 3px 0 rgba(0,0,0,.55); }
main { max-width: 920px; margin: 0 auto; padding: 20px 20px 150px; }
.card { position: relative; display: flex; gap: 16px; background: #151519; border: 1px solid #26262c; border-left: 6px solid var(--red); border-radius: 10px; padding: 18px 20px 16px; margin-bottom: 16px; opacity: 0; transform: translateY(30px) rotate(.4deg); animation: cardIn .65s cubic-bezier(.2,.8,.25,1) forwards; }
@keyframes cardIn { to { opacity: 1; transform: none; } }
.card.active { border-left-color: var(--yellow); background: #1d1d24; box-shadow: 0 12px 40px rgba(255,210,63,.14); }
.num { font-family: Georgia, 'Times New Roman', serif; font-size: 54px; font-weight: 700; color: #2c2c33; line-height: 1; }
.card.active .num { color: var(--yellow); }
.kicker { display: inline-block; font-size: 11px; font-weight: 800; letter-spacing: 2px; color: #0d0d0f; padding: 3px 9px; border-radius: 4px; margin-bottom: 8px; }
.ctitle { font-family: Georgia, 'Times New Roman', serif; font-size: 24px; line-height: 1.15; font-weight: 700; }
.card.active .ctitle { background: linear-gradient(transparent 55%, rgba(255,210,63,.85) 55%); }
.cmeta { color: var(--muted); font-size: 13px; margin-top: 8px; }
.cmeta a { color: var(--blue); }
.controls { position: sticky; bottom: 52px; display: flex; justify-content: center; align-items: center; gap: 16px; padding: 10px; z-index: 15; }
#playbtn { width: 64px; height: 64px; border-radius: 50%; border: none; background: var(--red); color: #fff; font-size: 22px; cursor: pointer; box-shadow: 0 8px 24px rgba(255,59,48,.4); transition: transform .15s; }
#playbtn:hover { transform: scale(1.06); }
.plabel { color: var(--muted); font-size: 13px; line-height: 1.45; }
.plabel b { color: var(--paper); font-variant-numeric: tabular-nums; }
details.digest { max-width: 920px; margin: 6px auto 30px; padding: 0 20px; }
details.digest summary { color: var(--yellow); font-weight: 700; cursor: pointer; letter-spacing: 1px; }
.digest-body { background: var(--paper); color: #161616; border-radius: 10px; padding: 18px 22px; margin-top: 12px; font-family: Georgia, 'Times New Roman', serif; }
.digest-body a { color: #0b5bd3; }
.digest-body h2 { font-size: 22px; }
.digest-body p { line-height: 1.55; }
.script { max-width: 920px; margin: 0 auto 70px; padding: 0 20px; color: var(--muted); font-size: 14px; line-height: 1.7; }
.made { max-width: 920px; margin: 0 auto; padding: 0 20px 90px; color: #5b6068; font-size: 12px; letter-spacing: 1px; }
.ticker { position: fixed; bottom: 0; left: 0; right: 0; display: flex; align-items: stretch; background: #000; border-top: 3px solid var(--red); z-index: 18; }
.ticklabel { background: var(--red); color: #fff; font-weight: 900; font-size: 12px; letter-spacing: 2px; padding: 12px 14px; display: flex; align-items: center; }
.tickwrap { overflow: hidden; flex: 1; display: flex; align-items: center; }
.ticker-inner { display: inline-block; white-space: nowrap; animation: marquee 45s linear infinite; color: #cfd3da; font-size: 13px; letter-spacing: 1px; padding: 10px 0; }
@keyframes marquee { from { transform: translateX(0); } to { transform: translateX(-50%); } }
"""

_ANIM_JS = """
const audio = document.getElementById('narration');
const cards = Array.from(document.querySelectorAll('.card'));
const bar = document.getElementById('pbar');
const btn = document.getElementById('playbtn');
const timeEl = document.getElementById('time');
const clockEl = document.getElementById('clock');
let lastActive = -1;
function fmt(t) { if (!isFinite(t)) return '0:00'; const m = Math.floor(t / 60), s = Math.floor(t % 60); return m + ':' + String(s).padStart(2, '0'); }
function tickClock() { clockEl.textContent = new Date().toLocaleTimeString(); }
tickClock(); setInterval(tickClock, 1000);
btn.addEventListener('click', () => { if (audio.paused) { audio.play(); } else { audio.pause(); } });
audio.addEventListener('play', () => { btn.textContent = '\\u275A\\u275A'; });
audio.addEventListener('pause', () => { btn.textContent = '\\u25B6'; });
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
audio.addEventListener('ended', () => { cards.forEach(c => c.classList.remove('active')); btn.textContent = '\\u25B6'; lastActive = -1; });
"""


def kicker_for(story: Story) -> tuple[str, str]:
    title = story.title.lower()
    domain = (story.domain or "").lower()
    tokens = set(title.split())

    def kw(*words: str) -> bool:
        return any(w in title for w in words)

    def tok(*words: str) -> bool:
        return any(w in tokens for w in words)

    if kw("vulnerab", "exploit", "escape", "pwn", "hack", "breach", "malware", "classifier", "cyber") and "cybertruck" not in title:
        return ("SECURITY", "#ff5c5c")
    if kw("llm", "gpt", "gemini", "claude", "opus", "anthropic", "openai", "model", "agent", "copilot") or tok("ai", "ai-driven") or domain in {"anthropic.com", "openai.com"}:
        return ("AI", "#8b7bff")
    if kw("acqui", "funding", "raises", "raise", "startup", "ipo", "valuation", "venture", "billion", "million"):
        return ("BUSINESS", "#ffb020")
    if kw("chip", "gpu", "silicon", "hardware", "semiconductor", "cpu") or tok("mac", "macs"):
        return ("HARDWARE", "#3fd0c9")
    if kw("github", "git ", "linux", "kernel", "postgres", "database", "storage", "compiler", "vm", "sdk", "uefi") or tok("dev"):
        return ("DEV & INFRA", "#5cb85c")
    if kw("google", "microsoft", "apple", "meta", "amazon", "tesla", "nvidia", "faa"):
        return ("BIG TECH", "#ff7a59")
    return ("TECH", "#4da3ff")


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
        label, color = kicker_for(s)
        title = html.escape(s.title)
        dom = html.escape(s.domain or "news.ycombinator.com")
        hn = html.escape(s.hn_url, quote=True)
        src_link = f' &middot; <a href="{html.escape(s.url, quote=True)}">source</a>' if s.url else ""
        cards_html.append(
            f'<div class="card" id="card-{i}" style="animation-delay:{i * 4.5:.1f}s">'
            f'<div class="num">{i + 1}</div>'
            f'<div class="cbody"><span class="kicker" style="background:{color}">{label}</span>'
            f'<div class="ctitle">{title}</div>'
            f'<div class="cmeta">{dom} &middot; {s.points} pts &middot; {s.num_comments} comments &middot; '
            f'<a href="{hn}">HN discussion</a>{src_link}</div></div></div>'
        )
    ticker = " &nbsp;&#8226;&nbsp; ".join(html.escape(s.domain or "news.ycombinator.com") for s in stories)
    ticker = ticker + " &nbsp;&#8226;&nbsp; " + ticker
    digest_body = markdown_to_html(markdown).removeprefix("<html><body>").removesuffix("</body></html>")
    n = len(stories)
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(subject)}</title><style>{_ANIM_CSS}</style></head><body>"
        "<div id=\"pwrap\"><div id=\"pbar\"></div></div>"
        "<header><div class=\"topbar\"><span class=\"logo\">HN<em>&#9679;</em>WIRE</span>"
        "<span class=\"clock\" id=\"clock\">--:--:--</span></div>"
        "<h1 class=\"masthead\">Tech <span class=\"accent\">Digest</span></h1>"
        f"<div class=\"deck\">{html.escape(subject)} &middot; {html.escape(date_str)} &middot; {n} stories &middot; press play for the narration</div>"
        f"<div class=\"sticker\">TOP {n} TODAY</div></header>"
        "<main>" + "".join(cards_html) + "</main>"
        "<div class=\"controls\"><button id=\"playbtn\" title=\"Play narration\">&#9654;</button>"
        "<div class=\"plabel\"><b id=\"time\">0:00 / 0:00</b><br>narration &middot; voxtral voice</div></div>"
        f"<audio id=\"narration\" preload=\"auto\" src=\"data:audio/mpeg;base64,{audio_b64}\"></audio>"
        "<details class=\"digest\"><summary>FULL DIGEST &amp; TRANSCRIPT</summary>"
        f"<div class=\"digest-body\">{digest_body}</div></details>"
        f"<div class=\"script\"><strong>Narration transcript</strong><br>{html.escape(script)}</div>"
        "<div class=\"made\">MADE BY HN-TECH-DIGEST &middot; ALL LINKS GO TO THE ORIGINAL SOURCES</div>"
        "<footer class=\"ticker\"><span class=\"ticklabel\">LATEST</span>"
        f"<div class=\"tickwrap\"><div class=\"ticker-inner\">{ticker}</div></div></footer>"
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


async def llm_chat_json(system: str, user: str, retries: int = 2) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
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
        except ValueError as exc:
            last_error = exc
    raise last_error  # type: ignore[misc]


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
async def fetch_front_page(source: str = "front_page") -> list[Story]:
    payload = await hn_get_json(hn_source_url(source))
    return parse_front_page(payload)


@workflows.activity(name="hn-select-tech-stories", start_to_close_timeout=timedelta(seconds=180))
async def select_tech_stories(stories: list[Story], count: int) -> list[Story]:
    candidates = filter_candidates(stories)
    system, user = build_selection_prompt(candidates, count)
    try:
        raw = await llm_chat_json(system, user)
    except Exception:
        raw = {"selected_ids": []}  # apply_selection falls back to top-N by engagement
    return apply_selection(raw, candidates, count)


def fallback_digest(stories: list[Story], date_str: str) -> dict[str, Any]:
    lines = [
        f"{i}. **{s.title}** — [source]({s.url or s.hn_url}) · [HN discussion]({s.hn_url}) ({s.points} pts, {s.num_comments} comments)"
        for i, s in enumerate(stories, 1)
    ]
    markdown = (
        f"# HN Tech Digest - {date_str}\n\n" + "\n".join(lines) + "\n\nAudio briefing attached (MP3, about 1-2 minutes).\n"
    )
    script = (
        f"Here is your Hacker News tech digest for {date_str}. "
        + " ".join(f"{s.title}." for s in stories)
        + " That's all for today. See you tomorrow."
    )
    return {"subject": f"HN Tech Digest - {date_str}", "markdown": markdown, "script": clamp_script(script)}


@workflows.activity(name="hn-build-digest", start_to_close_timeout=timedelta(seconds=180))
async def build_digest(stories: list[Story], date_str: str) -> dict[str, Any]:
    system, user = build_digest_prompt(stories, date_str)
    try:
        digest = await llm_chat_json(system, user)
    except Exception:
        digest = fallback_digest(stories, date_str)
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
        "One-shot digest: reads the latest Hacker News stories (or front page/best via the `source` input), selects the top tech-industry stories, "
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
        stories = await fetch_front_page(input.source)
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
