from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree

from .base import http_get_text


CRITIC_BLOG_USER_AGENT = "cultural-mood-tracker-critic/0.2"


@dataclass(frozen=True)
class CriticSource:
    name: str
    strategy: str
    url: str


SOURCES: tuple[CriticSource, ...] = (
    CriticSource(name="rogerebert", strategy="feed", url="https://www.rogerebert.com/feed"),
    CriticSource(name="indiewire", strategy="sitemap_index", url="https://www.indiewire.com/sitemap_index.xml"),
    CriticSource(name="vulture", strategy="sitemap_index", url="https://www.vulture.com/sitemap.xml"),
    CriticSource(name="slant", strategy="feed", url="https://www.slantmagazine.com/feed/"),
    CriticSource(name="slashfilm", strategy="sitemap_index", url="https://www.slashfilm.com/sitemap_index.xml"),
)


class _ArticleBodyParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._stack: list[str] = []
        self._capture_depth = 0
        self._skip_depth = 0
        self._texts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        classes = f"{attrs_dict.get('class', '')} {attrs_dict.get('id', '')}".lower()
        self._stack.append(tag)

        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return

        if tag in {"article", "main"} or any(
            token in classes
            for token in ("article", "post-content", "entry-content", "content-body", "article-body", "body-content")
        ):
            self._capture_depth += 1
            return

        if self._capture_depth and tag in {"p", "h2", "h3", "li", "blockquote"}:
            self._texts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._stack:
            self._stack.pop()
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if self._capture_depth and tag in {"article", "main"}:
            self._capture_depth -= 1
        if self._capture_depth and tag in {"p", "h2", "h3", "li", "blockquote"}:
            self._texts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self._capture_depth:
            return
        text = data.strip()
        if text:
            self._texts.append(text)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "\n".join(self._texts)).strip()


class _ScopedArticleParser(HTMLParser):
    def __init__(
        self,
        *,
        target_tokens: tuple[str, ...],
        skip_tags: tuple[str, ...] = ("script", "style", "noscript", "svg", "form", "button"),
        skip_class_tokens: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self._target_tokens = tuple(token.lower() for token in target_tokens)
        self._skip_tags = set(skip_tags)
        self._skip_class_tokens = tuple(token.lower() for token in skip_class_tokens)
        self._capture_depth = 0
        self._skip_depth = 0
        self._texts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        attrs_blob = " ".join(value for _, value in attrs if value).lower()
        classes = f"{attrs_dict.get('class', '')} {attrs_dict.get('id', '')}".lower()
        searchable = f"{classes} {attrs_blob}".strip()
        matched_target = any(token in searchable for token in self._target_tokens)
        matched_skip = any(token in searchable for token in self._skip_class_tokens)

        if self._skip_depth:
            self._skip_depth += 1
            return

        if matched_skip or (self._capture_depth and tag in self._skip_tags):
            self._skip_depth = 1
            return

        if self._capture_depth:
            self._capture_depth += 1
        elif matched_target:
            self._capture_depth = 1

        if self._capture_depth and tag in {"p", "h2", "h3", "h4", "li", "blockquote"}:
            self._texts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if self._capture_depth and tag in {"p", "h2", "h3", "h4", "li", "blockquote"}:
            self._texts.append("\n")
        if self._capture_depth:
            self._capture_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self._capture_depth:
            return
        text = data.strip()
        if text:
            self._texts.append(text)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "\n".join(self._texts)).strip()


def list_sources() -> tuple[CriticSource, ...]:
    return SOURCES


def get_source(source_name: str) -> CriticSource:
    for source in SOURCES:
        if source.name == source_name:
            return source
    raise KeyError(f"Unknown critic source: {source_name}")


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return parsed


def _parse_feed_timestamp(value: str | None) -> str | None:
    parsed = _parse_datetime(value)
    if not parsed:
        return None
    return parsed.isoformat().replace("+00:00", "Z")


def _within_window(published_at: str | None, start_date: str, end_date: str) -> bool:
    parsed = _parse_datetime(published_at)
    if not parsed:
        return True
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    return start <= parsed.date() <= end


def fetch_feed_entries(feed_url: str, *, entry_limit: int) -> list[dict[str, Any]]:
    payload = http_get_text(feed_url, user_agent=CRITIC_BLOG_USER_AGENT, timeout=60)
    root = ElementTree.fromstring(payload)
    entries: list[dict[str, Any]] = []

    for item in root.findall(".//item"):
        title = html.unescape("".join(item.findtext("title", default="")).strip())
        link = "".join(item.findtext("link", default="")).strip()
        description = html.unescape("".join(item.findtext("description", default="")).strip())
        author = item.findtext("{http://purl.org/dc/elements/1.1/}creator") or item.findtext("author")
        published_at = _parse_feed_timestamp(item.findtext("pubDate"))
        guid = item.findtext("guid") or link
        entries.append(
            {
                "title": title,
                "link": link,
                "description": description,
                "author": (author or "").strip() or None,
                "published_at": published_at,
                "guid": guid,
            }
        )
        if len(entries) >= entry_limit:
            break

    return entries


def _strip_tags(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(value))).strip()


def _slug_title_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return ""
    slug = path.split("/")[-1]
    slug = re.sub(r"\.html?$", "", slug)
    return re.sub(r"[-_]+", " ", slug).strip()


def _extract_json_ld_article_body(html_text: str) -> str:
    matches = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for raw_match in matches:
        try:
            payload = json.loads(html.unescape(raw_match))
        except json.JSONDecodeError:
            continue
        candidates = payload if isinstance(payload, list) else [payload]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            body = candidate.get("articleBody")
            if isinstance(body, str) and body.strip():
                return body.strip()
    return ""


def _looks_substantive(text: str, *, min_words: int = 80, min_chars: int = 500) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) >= min_chars:
        return True
    return len(normalized.split()) >= min_words


def _extract_vulture_article_body(html_text: str) -> str:
    parser = _ScopedArticleParser(
        target_tokens=("article-content", "vulture-zephr-anchor"),
        skip_tags=("script", "style", "noscript", "svg", "form", "button", "picture"),
        skip_class_tokens=(
            "newsletter",
            "related",
            "most-popular",
            "sign-in",
            "account-step",
            "comments",
            "comment",
            "recaptcha",
        ),
    )
    parser.feed(html_text)
    return parser.text()


def _extract_indiewire_article_body(html_text: str) -> str:
    parser = _ScopedArticleParser(
        target_tokens=("gutenberg-content__content",),
        skip_tags=("script", "style", "noscript", "svg", "form", "button", "picture"),
        skip_class_tokens=(
            "cardsrelatedcontent",
            "newsletter",
            "social-media",
            "adunit",
            "trinityaudioplaceholder",
            "pmc-recaptcha",
            "menu__heading-wrapper",
            "cards__heading-wrapper",
        ),
    )
    parser.feed(html_text)
    return parser.text()


def _extract_slashfilm_article_body(html_text: str) -> str:
    parser = _ArticleBodyParser()
    parser.feed(html_text)
    return parser.text()


def _postprocess_vulture_text(text: str) -> str:
    cleaned = text
    stop_markers = (
        "\nSign up for the Vulture Daily",
        "\nRelated\n",
        "\nTags:\n",
        "\nShow\nComment\n",
        "\nYour product is saved!",
        "\nMost Viewed Stories\n",
        "\nMost Popular\n",
        "\nLatest News from Vulture\n",
        "\nMore Stories\n",
        "\nSign In to Comment\n",
        "\nSign In To Continue Reading\n",
        "\nCreate Your Free Account\n",
    )
    cut_positions = [cleaned.find(marker) for marker in stop_markers if marker in cleaned]
    cut_positions = [position for position in cut_positions if position >= 0]
    if cut_positions:
        cleaned = cleaned[: min(cut_positions)]
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _postprocess_indiewire_text(text: str) -> str:
    cleaned = text
    stop_markers = (
        "\nRead More:\n",
        "\nDaily Headlines\n",
        "\nMore from IndieWire\n",
        "\nMust Read\n",
        "\nMore From IndieWire\n",
        "\nMost Popular\n",
        "\nYou may also like\n",
        "\nAbout\n",
        "\nNewsletter Sign Up\n",
        "\nHave a Tip?\n",
        "\nPMC Logo\n",
    )
    cut_positions = [cleaned.find(marker) for marker in stop_markers if marker in cleaned]
    cut_positions = [position for position in cut_positions if position >= 0]
    if cut_positions:
        cleaned = cleaned[: min(cut_positions)]
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _postprocess_slashfilm_text(text: str) -> str:
    cleaned = text
    stop_markers = (
        "\nRecommended\n",
        "\nRecommended",
    )
    cut_positions = [cleaned.find(marker) for marker in stop_markers if marker in cleaned]
    cut_positions = [position for position in cut_positions if position >= 0]
    if cut_positions:
        cleaned = cleaned[: min(cut_positions)]
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def fetch_article_text(url: str) -> str:
    html_text = http_get_text(url, user_agent=CRITIC_BLOG_USER_AGENT, timeout=60)
    netloc = urlparse(url).netloc
    if "vulture.com" in netloc:
        vulture_text = _extract_vulture_article_body(html_text)
        if vulture_text:
            return _postprocess_vulture_text(vulture_text)
    if "indiewire.com" in netloc:
        indiewire_text = _extract_indiewire_article_body(html_text)
        if indiewire_text:
            return _postprocess_indiewire_text(indiewire_text)
    if "slashfilm.com" in netloc:
        slashfilm_text = _extract_slashfilm_article_body(html_text)
        if slashfilm_text:
            return _postprocess_slashfilm_text(slashfilm_text)

    from_json_ld = _extract_json_ld_article_body(html_text)
    if from_json_ld and _looks_substantive(from_json_ld):
        return from_json_ld

    parser = _ArticleBodyParser()
    parser.feed(html_text)
    text = parser.text()
    if text:
        return text

    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html_text, flags=re.IGNORECASE | re.DOTALL)
    stripped = [_strip_tags(paragraph) for paragraph in paragraphs]
    return "\n\n".join(part for part in stripped if part)


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _collect_elements(node: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in node.iter() if _local_name(child.tag) == name]


def _find_child_text(node: ElementTree.Element, name: str) -> str | None:
    for child in node:
        if _local_name(child.tag) == name and child.text:
            return child.text.strip()
    return None


def _looks_like_article_url(source_name: str, url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path or path.endswith("/feed") or "/tag/" in path or "/author/" in path:
        return False

    parts = [part for part in path.split("/") if part]
    if source_name == "vulture":
        return parts[:1] == ["article"] and len(parts) >= 2
    if source_name == "slashfilm":
        if len(parts) < 2:
            return False
        if parts[0] == "category":
            return False
        if re.fullmatch(r"page-\d+(?:-\d+)?", parts[-1]):
            return False
        return parts[0].isdigit() or re.fullmatch(r"\d+", parts[0]) is not None
    if source_name == "indiewire":
        if len(parts) < 3:
            return False
        if parts[0] in {"c", "t", "page", "category", "author"}:
            return False
        return parts[-1].isdigit() or re.search(r"-\d{7,}$", parts[-1]) is not None
    return True


def _select_relevant_child_sitemaps(
    source_name: str,
    sitemaps: list[dict[str, str | None]],
    *,
    start_date: str,
    end_date: str,
) -> list[str]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    selected: list[str] = []

    for item in sitemaps:
        loc = item.get("loc") or ""
        lastmod = item.get("lastmod")
        if not loc:
            continue
        if source_name == "vulture":
            years = {str(year) for year in range(start.year, end.year + 1)}
            if any(f"sitemap-{year}.xml" in loc for year in years):
                selected.append(loc)
            continue
        if source_name == "indiewire":
            match = re.search(r"post-sitemap(\d{6})\.xml", loc)
            if not match:
                continue
            year = int(match.group(1)[:4])
            month = int(match.group(1)[4:])
            month_start = date(year, month, 1)
            month_end = date(year + (month // 12), (month % 12) + 1, 1) if month < 12 else date(year + 1, 1, 1)
            month_end = month_end.fromordinal(month_end.toordinal() - 1)
            if month_end >= start and month_start <= end:
                selected.append(loc)
            continue
        parsed_lastmod = _parse_datetime(lastmod)
        if parsed_lastmod and start <= parsed_lastmod.date() <= end:
            selected.append(loc)

    if source_name == "slashfilm" and not selected:
        selected = [item["loc"] for item in sitemaps[:8] if item.get("loc")]

    return selected


def fetch_sitemap_entries(
    source_name: str,
    sitemap_index_url: str,
    *,
    start_date: str,
    end_date: str,
    entry_limit: int,
) -> list[dict[str, Any]]:
    payload = http_get_text(sitemap_index_url, user_agent=CRITIC_BLOG_USER_AGENT, timeout=60)
    index_root = ElementTree.fromstring(payload)
    sitemap_nodes = [node for node in index_root.iter() if _local_name(node.tag) == "sitemap"]
    child_sitemaps = [
        {"loc": _find_child_text(node, "loc"), "lastmod": _find_child_text(node, "lastmod")}
        for node in sitemap_nodes
    ]
    sitemap_urls = _select_relevant_child_sitemaps(
        source_name,
        child_sitemaps,
        start_date=start_date,
        end_date=end_date,
    )

    max_entries = max(entry_limit * 50, 1500)
    entries: list[dict[str, Any]] = []
    seen_links: set[str] = set()

    for child_url in sitemap_urls:
        if len(entries) >= max_entries:
            break
        child_payload = http_get_text(child_url, user_agent=CRITIC_BLOG_USER_AGENT, timeout=60)
        child_root = ElementTree.fromstring(child_payload)
        url_nodes = [node for node in child_root.iter() if _local_name(node.tag) == "url"]
        for node in url_nodes:
            link = _find_child_text(node, "loc")
            if not link or link in seen_links or not _looks_like_article_url(source_name, link):
                continue
            published_at = _find_child_text(node, "lastmod") or _find_child_text(node, "publication_date")
            if not _within_window(published_at, start_date, end_date):
                continue
            title = _find_child_text(node, "title") or _slug_title_from_url(link)
            if not title:
                continue
            entries.append(
                {
                    "title": _strip_tags(title),
                    "link": link,
                    "description": None,
                    "author": None,
                    "published_at": _parse_feed_timestamp(published_at),
                    "guid": link,
                }
            )
            seen_links.add(link)
            if len(entries) >= max_entries:
                break

    return entries


def fetch_candidate_entries(
    source: CriticSource,
    *,
    start_date: str,
    end_date: str,
    entry_limit: int,
) -> list[dict[str, Any]]:
    if source.strategy == "feed":
        return [
            entry
            for entry in fetch_feed_entries(source.url, entry_limit=entry_limit)
            if _within_window(entry.get("published_at"), start_date, end_date)
        ]
    if source.strategy == "sitemap_index":
        return fetch_sitemap_entries(
            source.name,
            source.url,
            start_date=start_date,
            end_date=end_date,
            entry_limit=entry_limit,
        )
    raise RuntimeError(f"Unsupported critic source strategy: {source.strategy}")


def detect_document_type(*, content_type: str, headline: str, url: str) -> str:
    combined = f"{headline} {urlparse(url).path}".lower()
    if content_type == "tv" and "recap" in combined:
        return "tv_recap"
    if any(token in combined for token in ("explain", "ending", "analysis", "feature", "interview", "what-to-watch")):
        return "editorial_analysis"
    return "critic_review"
