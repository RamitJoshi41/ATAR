"""
M8 "live tools" search backend: real Wikipedia search for SEARCH_WEB.

Swapped into ATAREnv via ``search_backend=wikipedia_search`` when the demo's
live-tools toggle is on.  The policy was trained on M1's simulated BM25
search over Wikipedia abstracts, so live Wikipedia intro extracts are the
closest real-world counterpart — but still a different distribution, which
the UI must label.

Security (AGENTS.md rule 8 — fail closed):
  - one fixed HTTPS endpoint; the query only ever travels as an encoded
    URL parameter, so it cannot change the host or path;
  - query must be a non-empty string of at most MAX_QUERY_CHARS;
  - 5 s timeout, response size capped at MAX_RESPONSE_BYTES;
  - every failure (network, HTTP status, bad JSON, no result) returns a
    ToolResult with success=False; nothing raises across this boundary.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any, Callable

from atar.shared.shared_types import ToolResult

ENDPOINT = "https://en.wikipedia.org/w/api.php"
MAX_QUERY_CHARS = 300
MAX_RESPONSE_BYTES = 200_000
MAX_OUTPUT_CHARS = 1_000
TIMEOUT_S = 5.0
USER_AGENT = "ATAR-demo/0.1 (https://github.com/RamitJoshi41/ATAR; research demo)"

# urlopen-compatible callable; injectable so tests never touch the network.
Opener = Callable[..., Any]


def _fail(error: str, t0: float) -> ToolResult:
    return ToolResult(
        success=False, output=None, error=error,
        latency_ms=(time.monotonic() - t0) * 1000.0, token_cost=0,
    )


def build_url(query: str) -> str:
    """The one URL this module ever requests: search + intro extract of the top hit."""
    params = {
        "action": "query", "format": "json", "formatversion": "2",
        "generator": "search", "gsrsearch": query, "gsrlimit": "1",
        "prop": "extracts", "exintro": "1", "explaintext": "1", "redirects": "1",
    }
    return f"{ENDPOINT}?{urllib.parse.urlencode(params)}"


def wikipedia_search(params: dict, opener: Opener = urllib.request.urlopen) -> ToolResult:
    """SEARCH_WEB via Wikipedia: returns the intro of the best-matching article."""
    t0 = time.monotonic()
    query = params.get("query") if isinstance(params, dict) else None
    if not isinstance(query, str) or not query.strip():
        return _fail("Empty or non-string query", t0)
    if len(query) > MAX_QUERY_CHARS:
        return _fail(f"Query longer than {MAX_QUERY_CHARS} characters", t0)

    request = urllib.request.Request(build_url(query.strip()), headers={"User-Agent": USER_AGENT})
    try:
        with opener(request, timeout=TIMEOUT_S) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                return _fail(f"Wikipedia returned HTTP {status}", t0)
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except Exception as exc:  # network errors, timeouts, TLS — all fail closed
        return _fail(f"Search request failed: {type(exc).__name__}", t0)

    if len(body) > MAX_RESPONSE_BYTES:
        return _fail("Search response too large", t0)
    try:
        data = json.loads(body)
        pages = data["query"]["pages"]
        extract = str(pages[0].get("extract", "")).strip()
        title = str(pages[0].get("title", "")).strip()
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return _fail("No search result", t0)
    if not extract:
        return _fail("No search result", t0)

    output = f"{title}: {extract}" if title else extract
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS].rsplit(" ", 1)[0] + " …"
    return ToolResult(
        success=True, output=output, error=None,
        latency_ms=(time.monotonic() - t0) * 1000.0,
        token_cost=len(output.split()),
    )
