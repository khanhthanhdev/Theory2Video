import os
import sys
import json
from typing import Any
from dotenv import load_dotenv
load_dotenv()
import requests


def load_api_key() -> str:
    """
    Load CONTEXT7_API_KEY from environment, falling back to .env if present.
    """
    # Lazy import to avoid hard dependency if user doesn't use .env
    key = os.getenv("CONTEXT7_API_KEY")
    # First, try python-dotenv if the key is not already in env
    if not key:
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
            key = os.getenv("CONTEXT7_API_KEY")
        except Exception:
            # Fallback: attempt a minimal parse of a local .env
            env_path = os.path.join(os.getcwd(), ".env")
            if os.path.exists(env_path):
                try:
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            if line.startswith("CONTEXT7_API_KEY"):
                                # Handle formats like KEY=value or KEY="value"
                                parts = line.split("=", 1)
                                if len(parts) == 2:
                                    raw = parts[1].strip()
                                    if (raw.startswith('"') and raw.endswith('"')) or (
                                        raw.startswith("'") and raw.endswith("'")
                                    ):
                                        raw = raw[1:-1]
                                    key = raw
                                    break
                except Exception:
                    pass

    if not key:
        raise SystemExit(
            "CONTEXT7_API_KEY is not set. Add it to your environment or .env file."
        )
    return key


def get_docs(library: str, topic: str = "", tokens: int = 2000, resp_type: str = "json") -> Any:
    """
    Call Context7 Get Docs API for a given library path like 'vercel/next.js'.
    Returns parsed JSON (if type=json) or text.
    """
    api_key = load_api_key()
    base_url = "https://context7.com/api/v1"
    url = f"{base_url}/{library}"
    params = {"type": resp_type, "tokens": str(tokens)}
    if topic:
        params["topic"] = topic

    headers = {"Authorization": f"Bearer {api_key}"}

    resp = requests.get(url, headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    if resp_type == "json":
        return resp.json()
    return resp.text


def main(argv: list[str]) -> int:
    """
    Simple smoke test for Context7 Get Docs API.

    Usage:
      python tests/test_context7_docs.py [library] [topic] [tokens]

    Examples:
      python tests/test_context7_docs.py vercel/next.js ssr 5000
      python tests/test_context7_docs.py vercel/next.js
    """
    library = "websites/manim_community-en-stable"
    topic = "3d object"
    tokens = 5000

    if len(argv) > 1 and argv[1]:
        library = argv[1]
    if len(argv) > 2 and argv[2]:
        topic = argv[2]
    if len(argv) > 3 and argv[3]:
        try:
            tokens = int(argv[3])
        except ValueError:
            print("Invalid tokens value; defaulting to 5000", file=sys.stderr)
            tokens = 5000

    print(f"Requesting docs for '{library}' (topic='{topic}', tokens={tokens})...")
    raw = get_docs(library, topic=topic, tokens=tokens, resp_type="json")

    # Normalize to list shape across API variants: either [ ... ] or { snippets: [ ... ] }
    if isinstance(raw, dict) and "snippets" in raw and isinstance(raw["snippets"], list):
        data = raw["snippets"]
    elif isinstance(raw, list):
        data = raw
    else:
        print("Unexpected response shape. Raw:")
        print(json.dumps(raw, indent=2) if isinstance(raw, (dict, list)) else str(raw))
        return 2

    if len(data) == 0:
        print("Empty response list from Context7 API.")
        return 3

    first = data[0]
    required_keys = [
        "codeTitle",
        "codeDescription",
        "codeLanguage",
        "codeId",
        "pageTitle",
        "codeList",
    ]
    missing = [k for k in required_keys if k not in first]
    if missing:
        print(f"Missing expected keys in first item: {missing}")
        print("First item:")
        print(json.dumps(first, indent=2))
        return 4

    print(
        f"OK: Received {len(data)} snippet(s). First title: {first.get('codeTitle')}"
    )
    # Optionally, show a short preview of the first code block
    clist = first.get("codeList") or []
    if isinstance(clist, list) and clist:
        code_preview = clist[0].get("code", "")
        if isinstance(code_preview, str):
            print("Preview:\n" + (code_preview[:200] + ("..." if len(code_preview) > 200 else "")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
