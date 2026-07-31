"""Secure loopback API and browser host for Atom Harness Operator V4."""

from __future__ import annotations

import argparse
import json
import secrets
import signal
import threading
import webbrowser
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

from atom_harness_knowledge import (
    ATOM_HARNESS_RAG_RUNTIME,
    ATOM_HARNESS_WIKI_RUNTIME,
)
from atom_harness_operator import (
    ATOM_HARNESS_OPERATOR_RUNTIME,
    AtomHarnessOperator,
    OperatorCapacityError,
    OperatorStateError,
)
from atom_harness_operator_ui import (
    ATOM_HARNESS_OPERATOR_ARTIFACT_BINDING,
    ATOM_HARNESS_OPERATOR_UI_RUNTIME,
    render_operator_surface,
)
from atom_harness_session import AtomHarnessSession, default_session_output_root


ATOM_HARNESS_OPERATOR_SERVER_RUNTIME = "atom-harness-operator-loopback-server-v1"
LOOPBACK_HOST = "127.0.0.1"
MAX_REQUEST_BODY_BYTES = 16 * 1024


class AtomOperatorHTTPServer(ThreadingHTTPServer):
    """Own the operator and its in-memory browser access token."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        address: tuple[str, int],
        operator: AtomHarnessOperator,
    ) -> None:
        if address[0] != LOOPBACK_HOST:
            raise ValueError("Atom operator server must bind to IPv4 loopback")
        self.operator = operator
        self.access_token = secrets.token_urlsafe(32)
        self.shutdown_started = threading.Event()
        super().__init__(address, AtomOperatorRequestHandler)

    @property
    def origin(self) -> str:
        return f"http://{LOOPBACK_HOST}:{self.server_port}"

    @property
    def expected_host(self) -> str:
        return f"{LOOPBACK_HOST}:{self.server_port}"

    def initiate_shutdown(self, *, cancel_pending: bool) -> None:
        if self.shutdown_started.is_set():
            return
        self.shutdown_started.set()

        def finish() -> None:
            try:
                self.operator.shutdown(
                    wait=True,
                    cancel_pending=cancel_pending,
                )
            finally:
                self.shutdown()

        threading.Thread(
            target=finish,
            name="atom-harness-operator-shutdown",
            daemon=True,
        ).start()


class AtomOperatorRequestHandler(BaseHTTPRequestHandler):
    """Expose only typed, authenticated operator ramps on loopback."""

    server: AtomOperatorHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _host_valid(self) -> bool:
        return self.headers.get("Host", "") == self.server.expected_host

    def _authorized(self) -> bool:
        supplied = self.headers.get("X-Atom-Operator-Token", "")
        return bool(supplied) and secrets.compare_digest(
            supplied,
            self.server.access_token,
        )

    def _artifact_authorized(self) -> bool:
        if self._authorized():
            return True
        raw_cookie = self.headers.get("Cookie", "")
        if not raw_cookie or len(raw_cookie) > 4096:
            return False
        cookies = SimpleCookie()
        try:
            cookies.load(raw_cookie)
        except CookieError:
            return False
        morsel = cookies.get("AtomArtifactToken")
        return morsel is not None and secrets.compare_digest(
            morsel.value,
            self.server.access_token,
        )

    def _post_origin_valid(self) -> bool:
        return self.headers.get("Origin", "") == self.server.origin

    def _send_headers(
        self,
        *,
        status: int,
        content_type: str,
        content_length: int,
        nonce: str | None = None,
        frame_policy: str = "DENY",
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", frame_policy)
        self.send_header(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
        )
        if nonce is not None:
            self.send_header(
                "Content-Security-Policy",
                (
                    "default-src 'none'; "
                    f"style-src 'nonce-{nonce}'; "
                    f"script-src 'nonce-{nonce}'; "
                    "connect-src 'self'; "
                    "img-src 'self' data:; "
                    "frame-src 'self'; "
                    "font-src 'none'; object-src 'none'; base-uri 'none'; "
                    "form-action 'self'; frame-ancestors 'none'"
                ),
            )
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()

    def _bytes(
        self,
        payload: bytes,
        *,
        status: int = HTTPStatus.OK,
        content_type: str,
        nonce: str | None = None,
        frame_policy: str = "DENY",
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._send_headers(
            status=int(status),
            content_type=content_type,
            content_length=len(payload),
            nonce=nonce,
            frame_policy=frame_policy,
            extra_headers=extra_headers,
        )
        self.wfile.write(payload)

    def _json(
        self,
        payload: Mapping[str, Any],
        *,
        status: int = HTTPStatus.OK,
    ) -> None:
        raw = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        self._bytes(
            raw,
            status=status,
            content_type="application/json; charset=utf-8",
        )

    def _error(self, status: int, code: str) -> None:
        self._json(
            {"schema": 1, "error": code},
            status=status,
        )

    def _read_json(self) -> dict[str, Any]:
        media_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if media_type != "application/json":
            raise ValueError("content-type")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None or not raw_length.isdecimal():
            raise ValueError("content-length")
        length = int(raw_length)
        if not 0 <= length <= MAX_REQUEST_BODY_BYTES:
            raise OverflowError("request-body")
        data = self.rfile.read(length)
        if len(data) != length:
            raise ValueError("request-body")
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("json-object")
        return payload

    def _guard(self, *, post: bool = False) -> bool:
        if not self._host_valid():
            self._error(HTTPStatus.BAD_REQUEST, "invalid-host")
            return False
        if not self._authorized():
            self._error(HTTPStatus.UNAUTHORIZED, "authentication-required")
            return False
        if post and not self._post_origin_valid():
            self._error(HTTPStatus.FORBIDDEN, "invalid-origin")
            return False
        return True

    def do_GET(self) -> None:
        route = urlsplit(self.path).path
        if not self._host_valid():
            self._error(HTTPStatus.BAD_REQUEST, "invalid-host")
            return
        if route == "/":
            nonce = secrets.token_urlsafe(18)
            page = render_operator_surface(
                access_token=self.server.access_token,
                nonce=nonce,
            ).encode("utf-8")
            self._bytes(
                page,
                content_type="text/html; charset=utf-8",
                nonce=nonce,
                extra_headers={
                    "Set-Cookie": (
                        "AtomArtifactToken="
                        + self.server.access_token
                        + "; Path=/api/artifacts/; HttpOnly; SameSite=Strict"
                    )
                },
            )
            return
        if route == "/api/health":
            snapshot = self.server.operator.snapshot()
            self._json(
                {
                    "schema": 1,
                    "runtime": ATOM_HARNESS_OPERATOR_SERVER_RUNTIME,
                    "operator_runtime": ATOM_HARNESS_OPERATOR_RUNTIME,
                    "state": snapshot["state"],
                    "accepting": snapshot["accepting"],
                    "wiki_runtime": ATOM_HARNESS_WIKI_RUNTIME,
                    "rag_runtime": ATOM_HARNESS_RAG_RUNTIME,
                    "side_view_runtime": ATOM_HARNESS_OPERATOR_UI_RUNTIME,
                }
            )
            return
        if route == "/api/status":
            if not self._guard():
                return
            self._json(self.server.operator.snapshot())
            return
        prefix = "/api/artifacts/"
        suffix = "/side-view"
        if route.startswith(prefix) and route.endswith(suffix):
            if not self._artifact_authorized():
                self._error(HTTPStatus.UNAUTHORIZED, "authentication-required")
                return
            encoded = route[len(prefix) : -len(suffix)]
            request_id = unquote(encoded).strip("/")
            if (
                not request_id
                or "/" in request_id
                or "\\" in request_id
                or len(request_id) > 128
            ):
                self._error(HTTPStatus.BAD_REQUEST, "invalid-request-id")
                return
            try:
                path = self.server.operator.side_view_path(request_id)
                data = path.read_bytes()
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, "request-not-found")
                return
            except (OperatorStateError, ValueError, OSError):
                self._error(HTTPStatus.CONFLICT, "artifact-not-available")
                return
            self._bytes(
                data,
                content_type="text/html; charset=utf-8",
                frame_policy="SAMEORIGIN",
            )
            return
        if not self._guard():
            return
        self._error(HTTPStatus.NOT_FOUND, "route-not-found")

    def do_POST(self) -> None:
        if not self._guard(post=True):
            return
        route = urlsplit(self.path).path
        try:
            payload = self._read_json()
        except OverflowError:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request-too-large")
            return
        except (ValueError, UnicodeError, json.JSONDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "invalid-json-request")
            return
        try:
            if route == "/api/ask":
                if set(payload) != {"question"}:
                    raise ValueError("ask-fields")
                record = self.server.operator.submit(str(payload["question"]))
                self._json(record, status=HTTPStatus.ACCEPTED)
                return
            if route == "/api/cancel":
                if set(payload) != {"request_id"}:
                    raise ValueError("cancel-fields")
                record = self.server.operator.cancel(str(payload["request_id"]))
                self._json(record)
                return
            if route == "/api/retry":
                if set(payload) != {"request_id"}:
                    raise ValueError("retry-fields")
                record = self.server.operator.retry(str(payload["request_id"]))
                self._json(record, status=HTTPStatus.ACCEPTED)
                return
            if route == "/api/restart":
                if payload:
                    raise ValueError("restart-fields")
                result = self.server.operator.restart_resident_lane()
                self._json(result)
                return
            if route == "/api/shutdown":
                if set(payload) - {"cancel_pending"}:
                    raise ValueError("shutdown-fields")
                cancel_pending = payload.get("cancel_pending", False)
                if type(cancel_pending) is not bool:
                    raise ValueError("shutdown-cancel")
                self._json(
                    {
                        "schema": 1,
                        "accepted": True,
                        "cancel_pending": cancel_pending,
                    },
                    status=HTTPStatus.ACCEPTED,
                )
                self.server.initiate_shutdown(cancel_pending=cancel_pending)
                return
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "request-not-found")
            return
        except OperatorCapacityError:
            self._error(HTTPStatus.TOO_MANY_REQUESTS, "operator-capacity")
            return
        except OperatorStateError:
            self._error(HTTPStatus.CONFLICT, "operator-state-conflict")
            return
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "invalid-control-request")
            return
        self._error(HTTPStatus.NOT_FOUND, "route-not-found")


def build_server(
    operator: AtomHarnessOperator,
    *,
    port: int = 0,
) -> AtomOperatorHTTPServer:
    if not 0 <= int(port) <= 65535:
        raise ValueError("operator port is invalid")
    return AtomOperatorHTTPServer((LOOPBACK_HOST, int(port)), operator)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the persistent local Atom Harness Operator V4."
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--llama-server")
    parser.add_argument("--gpu-layers", default="auto")
    parser.add_argument("--provider-timeout-seconds", type=int, default=240)
    parser.add_argument("--startup-timeout-seconds", type=int)
    parser.add_argument("--lane-acquire-timeout-seconds", type=float)
    parser.add_argument("--max-queue-depth", type=int, default=8)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    default_output = default_session_output_root()
    output_root = (
        Path(arguments.output_root)
        if arguments.output_root is not None
        else default_output.with_name(
            default_output.name.replace(
                "resident-session",
                "operator-v4",
            )
        )
    ).resolve()
    session = AtomHarnessSession.official_local(
        output_root=output_root,
        model_path=arguments.model_path,
        llama_server=arguments.llama_server,
        gpu_layers=arguments.gpu_layers,
        provider_timeout_seconds=arguments.provider_timeout_seconds,
        startup_timeout_seconds=arguments.startup_timeout_seconds,
        lane_acquire_timeout_seconds=arguments.lane_acquire_timeout_seconds,
        max_queue_depth=arguments.max_queue_depth,
        max_concurrency=1,
    )
    operator = AtomHarnessOperator(
        session,
        state_root=output_root,
        max_queue_depth=arguments.max_queue_depth,
    )
    server = build_server(operator, port=arguments.port)

    def stop(_signal: int, _frame: Any) -> None:
        server.initiate_shutdown(cancel_pending=False)

    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)
    try:
        operator.start()
        startup = {
            "schema": 1,
            "runtime": ATOM_HARNESS_OPERATOR_SERVER_RUNTIME,
            "operator_runtime": ATOM_HARNESS_OPERATOR_RUNTIME,
            "origin": server.origin,
            "output_root": str(output_root),
            "wiki_runtime": ATOM_HARNESS_WIKI_RUNTIME,
            "rag_runtime": ATOM_HARNESS_RAG_RUNTIME,
            "side_view_runtime": ATOM_HARNESS_OPERATOR_UI_RUNTIME,
            "artifact_binding_marker": ATOM_HARNESS_OPERATOR_ARTIFACT_BINDING,
            "access_token_persisted": False,
            "cloud_allowed": False,
        }
        print(json.dumps(startup, sort_keys=True), flush=True)
        if not arguments.no_browser:
            webbrowser.open(server.origin, new=1)
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        operator.shutdown(wait=True, cancel_pending=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
