"""Fixtures shared by the unit and integration tests.

mmorch is imported inside fixture bodies, never at module level, so collecting the tests does not depend on any
one module of the package. The test directories have no __init__.py: pytest runs with --import-mode=importlib.

- repo_root, golden_dir, golden_hashes, golden_csv: the checkout, the goldens and the committed golden CSVs
- clean_env (autouse): no endpoint, model, matrix or OpenAI SDK variables leak in from the developer's shell
- mmorch_logger (autouse): the 'mmorch' logger, unconfigured for every test and restored afterwards
- fake_clock: the FakeClock class, a scripted clock for the timing code
- tiny_prompts, tiny_prompt_records: a four-prompt file in the released format
- llm_stub: a local OpenAI-compatible endpoint (StubLLM) that records every request
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import socketserver
import sys
import threading
from collections.abc import Callable, Iterable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, NamedTuple

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# Variables that change what the commands do; clean_env removes them for every test.
ENVIRONMENT_VARIABLES = (
    "LLM_API_BASE",
    "LLM_API_KEY",
    "MODEL_LOW",
    "MODEL_MEDIUM",
    "MODEL_HIGH",
    "MODEL_CLASSIFIER",
    "KUBERNETES_NAMESPACE",
    "MATRIX_CHART_DIR",
    "API_HOST",
    "API_PORT",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
)
LOCAL_HOSTS = ("127.0.0.1", "localhost")


# ---------------------------------------------------------------- checkout and goldens


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The checkout that holds tests/. Skips the test unless the released prompts and traces are present.

    An sdist ships tests/ but not data/ or results/, so the tests that need them skip there.
    """
    missing = [
        name
        for name, present in (
            ("data/prompts.jsonl.gz", (REPO_ROOT / "data" / "prompts.jsonl.gz").is_file()),
            ("results/traces/", (REPO_ROOT / "results" / "traces").is_dir()),
        )
        if not present
    ]
    if missing:
        pytest.skip(f"the released data is not available: {', '.join(missing)}")
    return REPO_ROOT


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    """tests/golden: the digests and console output of the v1.0.0 reproduction. It ships with the tests."""
    return TESTS_DIR / "golden"


@pytest.fixture(scope="session")
def golden_hashes(golden_dir: Path) -> dict[str, str]:
    """{path relative to results/: sha256} for the 13 reproduction outputs (8 CSVs, 5 PNGs).

    Parsed from tests/golden/reproduce_hashes.txt, whose lines are '<sha256> *<relative path>'; the '*' marks
    sha256sum's binary mode and is not part of the path. Figure paths keep their forward slash, e.g.
    'figures/fig4_complexity_distribution.png'.
    """
    hashes: dict[str, str] = {}
    for line in (golden_dir / "reproduce_hashes.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            hashes[name.removeprefix("*")] = digest
    return hashes


@pytest.fixture(scope="session")
def golden_csv(golden_hashes: dict[str, str]) -> Callable[[str], Path]:
    """golden_csv(name) -> results/<name>, the committed golden copy of a reproduction CSV, checked by sha256.

    tests/golden keeps only the digests; the eight golden CSVs themselves are the committed results/*.csv. A copy
    that no longer has its golden digest (for example after a local `mmorch reproduce` with broken code) fails
    the test instead of quietly becoming the reference. Skips when results/ is absent, as in an sdist.
    """

    def _golden_csv(name: str) -> Path:
        path = REPO_ROOT / "results" / name
        if not path.is_file():
            pytest.skip(f"the committed results/{name} is not available")
        if hashlib.sha256(path.read_bytes()).hexdigest() != golden_hashes[name]:
            pytest.fail(
                f"results/{name} no longer matches tests/golden/reproduce_hashes.txt; "
                f"restore it with `git checkout -- results/{name}`"
            )
        return path

    return _golden_csv


# ---------------------------------------------------------------- environment


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test without the developer's endpoint, model, matrix and OpenAI SDK variables.

    NO_PROXY and no_proxy gain 127.0.0.1 and localhost, so the stub servers are reached directly even behind a
    proxy. MMORCH_STRICT_FIGURES is left alone: it switches the strict figure checks on.
    """
    for name in ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    # On Windows both names are the same variable, so the second pass finds the hosts present already.
    for name in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(name, "")
        present = {host.strip() for host in current.split(",")}
        missing = [host for host in LOCAL_HOSTS if host not in present]
        if missing:
            monkeypatch.setenv(name, ",".join(filter(None, [current, *missing])))


# ---------------------------------------------------------------- logging


@pytest.fixture(autouse=True)
def mmorch_logger() -> Iterator[logging.Logger]:
    """The 'mmorch' logger, unconfigured for every test and put back as it was afterwards.

    mmorch.cli.main() configures it for the command line: one handler on the stderr of the moment, a level, and
    propagation off. Left in place, that handler would write into a closed capture buffer, and caplog, which
    listens on the root logger, would miss every mmorch.* record of the later tests. So each test starts with no
    handler, level NOTSET and propagation on; afterwards the handlers added during the test are closed and the
    previous handlers, level and propagation are restored.
    """
    logger = logging.getLogger("mmorch")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    yield logger
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


# ---------------------------------------------------------------- clock


class FakeClock:
    """A scripted clock: each call returns the next of the given values, and an extra call fails the test.

    The failure is raised with pytest.fail, which `except Exception` does not catch, so a surplus read inside the
    code under test cannot be swallowed and turned into an ordinary error result.
    """

    def __init__(self, *values: float) -> None:
        self.values = values
        self.reads = 0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            if self.reads >= len(self.values):
                pytest.fail(
                    f"the clock was read {self.reads + 1} times, but only {len(self.values)} values are scripted"
                )
            value = self.values[self.reads]
            self.reads += 1
        return value

    @property
    def remaining(self) -> int:
        """The number of scripted values not read yet."""
        return len(self.values) - self.reads


@pytest.fixture
def fake_clock() -> type[FakeClock]:
    """The FakeClock class: FakeClock(1000.0, 1000.5) returns 1000.0, then 1000.5, then fails the test."""
    return FakeClock


# ---------------------------------------------------------------- prompts


@pytest.fixture
def tiny_prompt_records() -> list[dict[str, str]]:
    """The four records that tiny_prompts writes, in file order."""
    return [
        {"qid": "HumanEval_1", "benchmark": "HumanEval", "question": "What is the capital of France?"},
        {
            "qid": "HumanEval_2",
            "benchmark": "HumanEval",
            "question": 'Prove that "naïve" proofs fail.\nRésumé: √2, 証明, Größe.',
        },
        {"qid": "HumanEval_3", "benchmark": "HumanEval", "question": "x" * 250},
        {"qid": "MBPP_1", "benchmark": "MBPP", "question": "Write a function."},
    ]


@pytest.fixture
def tiny_prompts(tmp_path: Path, tiny_prompt_records: list[dict[str, str]]) -> Path:
    """A prompts file in the released format (gzip JSON lines, CRLF, raw UTF-8) holding four prompts.

    - HumanEval_1 'What is the capital of France?': keyword tier LOW
    - HumanEval_2 'Prove that ...' with non-ASCII text, double quotes and an embedded newline: HIGH
    - HumanEval_3 'x' * 250: MEDIUM, and longer than the 200 characters a routing record keeps
    - MBPP_1 'Write a function.': another benchmark, filtered out of a HumanEval run
    """
    path = tmp_path / "prompts.jsonl.gz"
    text = "".join(json.dumps(record, ensure_ascii=False) + "\r\n" for record in tiny_prompt_records)
    with gzip.open(path, "wb") as f:
        f.write(text.encode("utf-8"))
    return path


# ---------------------------------------------------------------- OpenAI-compatible stub endpoint


class Headers(dict[str, str]):
    """Request headers keyed by lowercased name; lookups ignore case, so 'Authorization' finds 'authorization'."""

    def __init__(self, items: Iterable[tuple[str, str]]) -> None:
        super().__init__((name.lower(), value) for name, value in items)

    def __getitem__(self, name: str) -> str:
        return super().__getitem__(name.lower())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and super().__contains__(name.lower())

    def get(self, name: str, default: Any = None) -> Any:
        return super().get(name.lower(), default)


class RecordedRequest(NamedTuple):
    """One request the stub received: the path, the headers and the raw body bytes."""

    path: str
    headers: Headers
    body: bytes


class StubLLM:
    """A local OpenAI-compatible endpoint serving POST /v1/chat/completions, which records every request.

    Tests may change three attributes at any time:

    - reply: the message content of a non-streaming completion (default 'MEDIUM');
    - stream_pieces: the content deltas of a streaming completion (default ('Hello', ' world'));
    - status: the HTTP status (default 200); any other status answers {'error': {'message': 'forced', 'type':
      'stub'}}.

    A non-streaming completion reports usage 5/1/6 (prompt/completion/total tokens). A request with stream=true
    gets chunked text/event-stream frames 'data: {chunk}\\n\\n': one delta per piece, then a chunk with no choices
    and usage 7/2/9, then 'data: [DONE]'. The server speaks HTTP/1.1 (keep-alive) on 127.0.0.1 and an ephemeral
    port, from a daemon thread.
    """

    def __init__(self) -> None:
        self.reply = "MEDIUM"
        self.stream_pieces: tuple[str, ...] = ("Hello", " world")
        self.status = 200
        self._requests: list[RecordedRequest] = []
        self._lock = threading.Lock()
        self._server = _StubServer(self)
        # A short poll interval keeps stop(), which waits for the serving loop to notice, fast.
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, name="llm-stub", daemon=True
        )

    @property
    def base_url(self) -> str:
        """The OpenAI base URL of the stub, 'http://127.0.0.1:<port>/v1'."""
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}/v1"

    @property
    def requests(self) -> list[RecordedRequest]:
        """A copy of the requests received so far, in arrival order."""
        with self._lock:
            return list(self._requests)

    def json_bodies(self) -> list[Any]:
        """The request bodies parsed as JSON, in arrival order."""
        return [json.loads(request.body) for request in self.requests]

    def clear(self) -> None:
        """Forget the requests received so far."""
        with self._lock:
            self._requests.clear()

    def start(self) -> None:
        """Start serving in the background."""
        self._thread.start()

    def stop(self) -> None:
        """Stop serving and close the listening socket."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def record(self, request: RecordedRequest) -> None:
        """Append a received request; called from the server's handler threads."""
        with self._lock:
            self._requests.append(request)


class _StubServer(ThreadingHTTPServer):
    daemon_threads = True
    # Clients keep connections alive in their pools; closing the server must not wait for them.
    block_on_close = False
    # The baseline runner opens 20 connections at once; the default backlog of 5 can refuse some on Windows.
    request_queue_size = 128

    def __init__(self, stub: StubLLM) -> None:
        self.stub = stub
        super().__init__(("127.0.0.1", 0), _StubHandler)

    def server_bind(self) -> None:
        # HTTPServer.server_bind also resolves the host name with socket.getfqdn, a reverse DNS lookup that can
        # take seconds on some machines; the stub does not need it.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = port

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A client dropping a kept-alive connection (a reset on Windows) is normal; anything else is a stub bug.
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


class _StubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30  # seconds an idle keep-alive connection may hold a handler thread
    server: _StubServer

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        stub = self.server.stub
        stub.record(RecordedRequest(self.path, Headers(self.headers.items()), body))
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": f"unknown path {self.path}", "type": "stub"}})
        elif stub.status != 200:
            self._send_json(stub.status, {"error": {"message": "forced", "type": "stub"}})
        else:
            request = json.loads(body)
            model = request.get("model", "")
            if request.get("stream"):
                self._send_stream(model, stub.stream_pieces)
            else:
                self._send_json(200, _completion(model, stub.reply))

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the per-request log lines off stderr."""

    def _send_json(self, status: int, payload: object) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_stream(self, model: str, pieces: Iterable[str]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for piece in pieces:
            delta = {"index": 0, "delta": {"content": piece}, "finish_reason": None}
            self._send_event(json.dumps(_chunk(model, [delta])))
        usage = {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
        self._send_event(json.dumps({**_chunk(model, []), "usage": usage}))
        self._send_event("[DONE]")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _send_event(self, data: str) -> None:
        frame = f"data: {data}\n\n".encode()
        self.wfile.write(f"{len(frame):X}\r\n".encode("ascii") + frame + b"\r\n")
        self.wfile.flush()


def _chunk(model: str, choices: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": "chatcmpl-stub", "object": "chat.completion.chunk", "created": 0, "model": model, "choices": choices}


def _completion(model: str, content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }


@pytest.fixture
def llm_stub() -> Iterator[StubLLM]:
    """A running StubLLM; point LLM_API_BASE at llm_stub.base_url. It is stopped after the test."""
    stub = StubLLM()
    stub.start()
    try:
        yield stub
    finally:
        stub.stop()
