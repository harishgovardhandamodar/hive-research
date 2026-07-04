from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import requests

from .config import Config
from .llm import LLMInterface
from .graph import KnowledgeGraph
from .pipeline import _sanitize_id

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"https?://[^\s<>\"']+")

IMAGE_PATTERN = re.compile(
    r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE
)

LINK_PATTERN = re.compile(
    r'<a[^>]+href=["\'](https?://[^"\']+)["\']', re.IGNORECASE
)


def extract_title(html: str) -> str:
    m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def extract_description(html: str) -> str:
    patterns = [
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']twitter:description["\'][^>]+content=["\']([^"\']+)["\']',
    ]
    for pat in patterns:
        m = re.search(pat, html, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def extract_text_content(html: str) -> str:
    text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:5000]


def extract_images(html: str, base_url: str) -> list[str]:
    urls = []
    for m in IMAGE_PATTERN.finditer(html):
        src = m.group(1)
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            parsed = requests.utils.urlparse(base_url)
            src = f"{parsed.scheme}://{parsed.netloc}{src}"
        elif not src.startswith("http"):
            continue
        urls.append(src)
    return urls[:5]


def extract_links(html: str, base_url: str) -> list[dict[str, str]]:
    results = []
    seen = set()
    for m in LINK_PATTERN.finditer(html):
        href = m.group(1)
        if href not in seen:
            seen.add(href)
            results.append({"url": href, "title": ""})
    return results[:10]


class WebIngester:
    def __init__(self, config: Config, llm: LLMInterface, kg: KnowledgeGraph) -> None:
        self.config = config
        self.llm = llm
        self.kg = kg

    def ingest(self, url: str) -> dict[str, Any]:
        existing = self.kg.get_paper(url)
        if existing:
            return {"status": "exists", "node_id": url}

        try:
            resp = requests.get(
                url,
                timeout=30,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
            )
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            return {"status": "error", "message": f"Failed to fetch URL: {e}"}

        title = extract_title(html) or url
        description = extract_description(html)
        body_text = extract_text_content(html)
        images = extract_images(html, url)
        links = extract_links(html, url)

        content_for_llm = (
            f"Title: {title}\n"
            f"{'Description: ' + description + chr(10) if description else ''}"
            f"Content: {body_text[:3000]}"
        )

        analysis = self.llm.extract_structured(
            f"Summarize this web page as JSON.\n\n{content_for_llm}\n\n"
            'Respond JSON only:\n'
            '{"summary": "2-3 sentence summary of the article",\n'
            ' "notes": "1-2 paragraph study notes explaining key ideas and findings",\n'
            ' "tags": ["keyword1", "keyword2", "keyword3"],\n'
            ' "concepts": [{"name":"KeyConcept", "definition":"Brief definition"}]}'
        )

        tags = analysis.get("tags", [])
        concepts_data = analysis.get("concepts", [])
        summary = analysis.get("summary", description)
        notes = analysis.get("notes", "")

        node_id = f"web_{_sanitize_id(title)[:40]}"

        web_node = self.kg.add_paper(
            paper_id=node_id,
            title=title[:120],
            authors="",
            published="",
            abstract=summary[:500],
            categories=[],
            affiliations=f"URL: {url}",
        )
        web_node.type = "web"

        # Download and save images
        vault_dir = Path(self.config.vault_dir)
        safe_title = _sanitize_id(title)[:50]
        web_vault = vault_dir / safe_title
        web_vault.mkdir(parents=True, exist_ok=True)
        figs_dir = web_vault / "figures"
        figs_dir.mkdir(parents=True, exist_ok=True)

        saved_images = []
        for idx, img_url in enumerate(images):
            try:
                ir = requests.get(img_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                if ir.status_code == 200 and len(ir.content) > 2048:
                    ext = Path(img_url.split("?")[0]).suffix or ".jpg"
                    fname = f"web_img_{idx+1:02d}{ext}"
                    (figs_dir / fname).write_bytes(ir.content)
                    saved_images.append({"filename": fname, "url": img_url})
            except Exception as e:
                logger.warning("Failed to download image %s: %s", img_url, e)

        # Write vault note
        if notes or saved_images:
            note_lines = [
                "---",
                f"url: {url}",
                f"title: \"{title}\"",
                f"tags: [{', '.join(tags)}]",
                "---",
                "",
            ]
            if summary:
                note_lines.extend(["## Summary", "", summary, ""])
            if notes:
                note_lines.extend(["## Notes", "", notes, ""])
            if saved_images:
                note_lines.extend(["", "## Figures", ""])
                for img in saved_images:
                    note_lines.extend([
                        f"![{img['filename']}](figures/{img['filename']})",
                        "",
                    ])
            (web_vault / "00_notes.md").write_text("\n".join(note_lines))

        stored_images = "; ".join(img["url"] for img in saved_images) if saved_images else "; ".join(images)
        stored_links = "; ".join(l["url"] for l in links) if links else ""
        web_node.definition = (
            f"Web resource\nURL: {url}\n"
            f"{'Images: ' + stored_images + chr(10) if stored_images else ''}"
            f"{'Links: ' + stored_links if stored_links else ''}"
        )

        for tag in tags:
            tid = _sanitize_id(tag)
            matched = self.kg.find_similar_concept(tag)
            if matched:
                cid = matched.id
            else:
                cid = tid
                self.kg.add_concept(cid, tag, concept_type="tag", definition=f"A web resource tagged with '{tag}'.")
            self.kg.add_edge(node_id, cid, "related_to")

        for c in concepts_data:
            label = c.get("name", "")
            if not label:
                continue
            cid = _sanitize_id(label)
            definition = c.get("definition", "")
            matched = self.kg.find_similar_concept(label)
            if matched:
                cid = matched.id
            else:
                self.kg.add_concept(cid, label, definition=definition, concept_type="concept")
            self.kg.add_edge(node_id, cid, "related_to")

        self.kg.save()

        return {
            "status": "added",
            "node_id": node_id,
            "title": title,
            "summary": summary,
            "tags": tags,
            "concepts": len(concepts_data),
            "images": len(saved_images),
            "links": len(links),
            "notes": bool(notes),
        }
