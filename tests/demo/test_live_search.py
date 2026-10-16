"""
Live Wikipedia search (M8 live-tools toggle) — fail-closed behaviour.
The network is never touched: a fake opener stands in for urlopen.
"""

from __future__ import annotations

import io
import json
import urllib.parse

import pytest

from atar.demo.live_search import ENDPOINT, MAX_QUERY_CHARS, MAX_RESPONSE_BYTES, build_url, wikipedia_search


class _Resp(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200) -> None:
        super().__init__(body)
        self.status = status


def _opener(body: bytes, status: int = 200, seen: list | None = None):
    def open_(request, timeout):  # noqa: ANN001 - urlopen signature
        if seen is not None:
            seen.append((request, timeout))
        return _Resp(body, status)
    return open_


def _page(title: str, extract: str) -> bytes:
    return json.dumps({"query": {"pages": [{"title": title, "extract": extract}]}}).encode()


def test_success_returns_title_and_intro() -> None:
    seen: list = []
    r = wikipedia_search({"query": "capital of France"},
                         opener=_opener(_page("Paris", "Paris is the capital of France."), seen=seen))
    assert r.success and r.error is None
    assert r.output == "Paris: Paris is the capital of France."
    request, timeout = seen[0]
    assert timeout == 5.0 and request.full_url.startswith(ENDPOINT)


@pytest.mark.parametrize("params", [{}, {"query": ""}, {"query": "   "}, {"query": 42}, {"query": None}, "not a dict"])
def test_bad_queries_are_rejected_without_a_request(params: object) -> None:
    seen: list = []
    r = wikipedia_search(params, opener=_opener(b"{}", seen=seen))  # type: ignore[arg-type]
    assert not r.success and seen == []


def test_overlong_query_rejected_without_a_request() -> None:
    seen: list = []
    r = wikipedia_search({"query": "x" * (MAX_QUERY_CHARS + 1)}, opener=_opener(b"{}", seen=seen))
    assert not r.success and "longer" in (r.error or "") and seen == []


@pytest.mark.parametrize("query", [
    "x&action=delete&title=Main_Page",
    "../../etc/passwd",
    "https://evil.example.com/?q=1",
    "a\r\nHost: evil.example.com",
])
def test_query_cannot_change_host_path_or_parameters(query: str) -> None:
    url = urllib.parse.urlparse(build_url(query))
    assert (url.scheme, url.netloc, url.path) == ("https", "en.wikipedia.org", "/w/api.php")
    params = urllib.parse.parse_qs(url.query)
    assert params["gsrsearch"] == [query]
    assert params["action"] == ["query"]           # the injected action=delete stayed inside gsrsearch


def test_http_error_status_fails_closed() -> None:
    r = wikipedia_search({"query": "x"}, opener=_opener(b"", status=503))
    assert not r.success and "503" in (r.error or "")


def test_network_exception_fails_closed() -> None:
    def boom(request, timeout):  # noqa: ANN001
        raise TimeoutError("slow")
    r = wikipedia_search({"query": "x"}, opener=boom)
    assert not r.success and "TimeoutError" in (r.error or "")


def test_oversized_response_fails_closed() -> None:
    r = wikipedia_search({"query": "x"}, opener=_opener(b"a" * (MAX_RESPONSE_BYTES + 10)))
    assert not r.success and "too large" in (r.error or "")


@pytest.mark.parametrize("body", [b"not json", b"{}", b'{"query": {"pages": []}}',
                                  _page("Empty", ""), b'{"query": {"pages": "oops"}}'])
def test_malformed_or_empty_results_fail_closed(body: bytes) -> None:
    r = wikipedia_search({"query": "x"}, opener=_opener(body))
    assert not r.success and r.error == "No search result"


def test_long_extract_is_truncated() -> None:
    r = wikipedia_search({"query": "x"}, opener=_opener(_page("T", "word " * 1000)))
    assert r.success and len(r.output) <= 1_002 and r.output.endswith("…")
