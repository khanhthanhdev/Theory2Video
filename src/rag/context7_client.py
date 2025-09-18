"""Context7 API helper for retrieving documentation snippets."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import requests


def _load_context7_api_key() -> Optional[str]:
    """Resolve the Context7 API key from environment or a local .env file."""
    key = os.getenv("CONTEXT7_API_KEY")
    if key:
        return key

    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
        key = os.getenv("CONTEXT7_API_KEY")
        if key:
            return key
    except Exception:
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("CONTEXT7_API_KEY"):
                            _, _, raw = line.partition("=")
                            raw = raw.strip()
                            if (raw.startswith('"') and raw.endswith('"')) or (
                                raw.startswith("'") and raw.endswith("'")
                            ):
                                raw = raw[1:-1]
                            return raw or None
            except Exception:
                pass

    return None


class Context7Client:
    """Thin wrapper around the Context7 Get Docs API."""

    def __init__(
        self,
        library: Optional[str] = None,
        base_url: str = "https://context7.com/api/v1",
        default_tokens: int = 2000,
        max_queries: int = 3,
        snippets_per_query: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.library = (library or os.getenv("CONTEXT7_LIBRARY") or "websites/manim_community-en-stable").strip("/")
        self.default_tokens = default_tokens
        self.max_queries = max_queries
        self.snippets_per_query = snippets_per_query
        self.api_key = _load_context7_api_key()
        self._cache: Dict[Tuple[str, int], List[Dict]] = {}

    @property
    def available(self) -> bool:
        """Return True when an API key is configured."""
        return bool(self.api_key)

    def fetch_snippets(self, query: str, tokens: Optional[int] = None) -> List[Dict]:
        """Fetch documentation snippets for a given query string."""
        query = (query or "").strip()
        if not query or not self.available:
            return []

        token_budget = tokens if tokens is not None else self.default_tokens
        cache_key = (query, token_budget)

        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{self.base_url}/{self.library}"
        params = {
            "type": "json",
            "tokens": str(token_budget),
            "topic": query,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure path
            print(f"Context7 request failed for query '{query}': {exc}")
            return []

        try:
            data = response.json()
        except ValueError:  # pragma: no cover - unexpected response shape
            print(f"Context7 response was not JSON for query '{query}': {response.text[:200]}...")
            return []

        if isinstance(data, dict) and isinstance(data.get("snippets"), list):
            snippets = data["snippets"]
        elif isinstance(data, list):
            snippets = data
        else:
            print(f"Context7 response has unexpected shape for query '{query}': {json.dumps(data)[:200]}...")
            return []

        # Trim to manageable size per query
        trimmed = snippets[: self.snippets_per_query]
        self._cache[cache_key] = trimmed
        return trimmed

    def format_snippets_markdown(self, results: List[Dict[str, object]]) -> str:
        """Convert aggregated snippet results to markdown suitable for LLM context."""
        if not results:
            return "No relevant documentation found via Context7."

        lines: List[str] = ["## Relevant Documentation (Context7)", ""]

        for entry in results:
            query = entry.get("query", "") if isinstance(entry, dict) else ""
            if query:
                lines.append(f"### Query: {query}")
            snippets = entry.get("snippets") if isinstance(entry, dict) else None
            if not isinstance(snippets, list):
                lines.append("No snippets returned for this query.\n")
                continue

            for snippet in snippets:
                if not isinstance(snippet, dict):
                    continue
                title = snippet.get("codeTitle") or snippet.get("pageTitle") or "Result"
                description = snippet.get("codeDescription") or snippet.get("description")
                link = snippet.get("link")

                lines.append(f"**{title}**")
                if description:
                    lines.append(str(description))
                if link:
                    lines.append(f"Source: {link}")

                code_blocks = snippet.get("codeList") if isinstance(snippet.get("codeList"), list) else []
                if code_blocks:
                    first_block = code_blocks[0]
                    if isinstance(first_block, dict):
                        language = first_block.get("codeLanguage") or "text"
                        code = first_block.get("code") or ""
                        if isinstance(code, str) and code.strip():
                            preview = code.strip()
                            if len(preview) > 500:
                                preview = preview[:500] + "\n..."
                            lines.append(f"```{language}\n{preview}\n```")

                lines.append("")

            lines.append("")

        return "\n".join(lines).strip()


__all__ = ["Context7Client", "_load_context7_api_key"]

