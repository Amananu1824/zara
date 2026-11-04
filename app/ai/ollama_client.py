"""Validated HTTP boundary for Ollama chat and streaming."""
from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Iterator

import requests

from app.core.config import settings
from threading import Event, Lock


class OllamaError(RuntimeError):
    """Normalized Ollama error with machine-readable metadata."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "ollama_error",
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status_code = status_code

def _reject_duplicate_json_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"Duplicate JSON key in Ollama response: {key}"
            )

        result[key] = value

    return result


class OllamaStreamCancelled(OllamaError):
    """The active Ollama operation was cancelled."""

    def __init__(self) -> None:
        super().__init__(
            "The active Ollama request was cancelled.",
            code="cancelled",
            retryable=False,
        )

class OllamaClient:
    _METRIC_FIELDS = (
        "done_reason",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_cached_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    )

    _RETRYABLE_HTTP_STATUS = {
        408, 425, 429, 500, 502, 503, 504,
    }

    _ALLOWED_ROLES = {
        "system",
        "user",
        "assistant",
        "tool",
    }

    _MAX_JSON_RESPONSE_BYTES = 16 * 1024 * 1024

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.base_url = (
            base_url or settings.ollama_base_url
        ).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout_seconds = (
            settings.ollama_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        self._active_stream_lock = Lock()
        self._active_stream_response: requests.Response | None = None
        self._cancel_requested = False

        if not self.base_url:
            raise ValueError("Ollama base URL cannot be empty.")

        if not self.model.strip():
            raise ValueError("Ollama model cannot be empty.")

        if self.timeout_seconds <= 0:
            raise ValueError("Ollama timeout must be positive.")

        self.session = requests.Session()
        self.last_chat_metrics: dict[str, Any] | None = None
        self.last_stream_metrics: dict[str, Any] | None = None

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    @classmethod
    def _validate_request(
        cls,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> None:
        if not isinstance(messages, list) or not messages:
            raise OllamaError(
                "Chat messages must be a non-empty list.",
                code="invalid_request",
            )

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise OllamaError(
                    f"Message {index} must be an object.",
                    code="invalid_request",
                )

            role = message.get("role")
            content = message.get("content")

            if role not in cls._ALLOWED_ROLES:
                raise OllamaError(
                    f"Message {index} has an unsupported role.",
                    code="invalid_request",
                )

            if not isinstance(content, str):
                raise OllamaError(
                    f"Message {index} content must be text.",
                    code="invalid_request",
                )

        if not isinstance(options, dict):
            raise OllamaError(
                "Generation options must be an object.",
                code="invalid_request",
            )

    @staticmethod
    def _validate_chat_response(
        data: Any,
    ) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise OllamaError(
                "Ollama returned an invalid response object.",
                code="invalid_response",
            )

        error = data.get("error")
        if error:
            raise OllamaError(
                str(error),
                code="model_error",
            )

        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaError(
                "Ollama response is missing its message object.",
                code="invalid_response",
            )

        role = message.get("role")
        content = message.get("content")

        if role not in (None, "assistant"):
            raise OllamaError(
                "Ollama returned an unexpected message role.",
                code="invalid_response",
            )

        if not isinstance(content, str):
            raise OllamaError(
                "Ollama response content must be text.",
                code="invalid_response",
            )

        tool_calls = message.get("tool_calls")

        if tool_calls is not None and not isinstance(tool_calls, list):
            raise OllamaError(
                "Ollama returned malformed tool calls.",
                code="invalid_response",
            )

        return data

    def chat(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        format: str | dict[str, Any] | None = None,
        cancel_event: Event | None = None,
    ) -> dict[str, Any]:
        self._validate_request(messages, options)

        self.last_chat_metrics = None
        self.last_stream_metrics = None

        if cancel_event is not None and cancel_event.is_set():
            raise OllamaStreamCancelled()

        with self._active_stream_lock:
            self._cancel_requested = False

        started = perf_counter()

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": options,
        }

        if format is not None:
            if format != "json" and not isinstance(format, dict):
                raise OllamaError(
                    "Format must be 'json' or a JSON schema.",
                    code="invalid_request",
                )

            payload["format"] = format

        if tools is not None:
            if not isinstance(tools, list) or not tools:
                raise OllamaError(
                    "Tools must be a non-empty application-owned list.",
                    code="invalid_request",
                )

            payload["tools"] = tools

        response: requests.Response | None = None

        try:
            # HTTP response is read incrementally for cancellation and
            # response-size enforcement; model output remains non-streaming.
            response = self.session.post(
                self.chat_url,
                json=payload,
                stream=True,
                timeout=(10, self.timeout_seconds),
            )

            with self._active_stream_lock:
                self._active_stream_response = response
                was_cancelled = self._cancel_requested

            if was_cancelled or (
                cancel_event is not None and cancel_event.is_set()
            ):
                raise OllamaStreamCancelled()

            if not response.ok:
                self._raise_http_error(response)

            body = bytearray()

            for chunk in response.iter_content(
                chunk_size=64 * 1024
            ):
                if self._cancel_requested or (
                    cancel_event is not None and cancel_event.is_set()
                ):
                    raise OllamaStreamCancelled()

                if perf_counter() - started > self.timeout_seconds:
                    raise OllamaError(
                        "Ollama request exceeded its timeout.",
                        code="timeout",
                        retryable=True,
                    )

                if not chunk:
                    continue

                body.extend(chunk)

                if len(body) > self._MAX_JSON_RESPONSE_BYTES:
                    raise OllamaError(
                        "Ollama response exceeded the allowed size.",
                        code="response_too_large",
                    )

            if self._cancel_requested or (
                cancel_event is not None and cancel_event.is_set()
            ):
                raise OllamaStreamCancelled()

            try:
                raw_data = json.loads(
                    body.decode("utf-8-sig"),
                    object_pairs_hook=_reject_duplicate_json_pairs,
                )
            except (ValueError, UnicodeError) as exc:
                raise OllamaError(
                    "Ollama returned invalid JSON.",
                    code="invalid_json",
                ) from exc

            data = self._validate_chat_response(raw_data)

            metrics = {
                key: data[key]
                for key in self._METRIC_FIELDS
                if key in data
            }

            metrics.update({
                "model": data.get("model", self.model),
                "request_wall_seconds": perf_counter() - started,
                "stream_completed": True,
            })

            self.last_chat_metrics = metrics
            return data

        except OllamaError:
            raise

        # except requests.exceptions.Timeout as exc:
        #     if cancel_event is not None and cancel_event.is_set():
        #         raise OllamaStreamCancelled() from exc

        #     raise OllamaError(
        #         "Ollama request timed out.",
        #         code="timeout",
        #         retryable=True,
        #     ) from exc

        # except requests.exceptions.ConnectionError as exc:
        #     if cancel_event is not None and cancel_event.is_set():
        #         raise OllamaStreamCancelled() from exc

        #     raise OllamaError(
        #         "Could not connect to Ollama.",
        #         code="connection_error",
        #         retryable=True,
        #     ) from exc
        except requests.exceptions.Timeout as exc:
            with self._active_stream_lock:
                cancelled = self._cancel_requested

            if cancelled or (
                cancel_event is not None and cancel_event.is_set()
            ):
                raise OllamaStreamCancelled() from exc

            raise OllamaError(
                "Ollama request timed out.",
                code="timeout",
                retryable=True,
            ) from exc

        except requests.exceptions.ConnectionError as exc:
            with self._active_stream_lock:
                cancelled = self._cancel_requested

            if cancelled or (
                cancel_event is not None and cancel_event.is_set()
            ):
                raise OllamaStreamCancelled() from exc

            raise OllamaError(
                "Could not connect to Ollama.",
                code="connection_error",
                retryable=True,
            ) from exc
        
        except requests.exceptions.RequestException as exc:
            with self._active_stream_lock:
                cancelled = self._cancel_requested

            if cancelled or (
                cancel_event is not None and cancel_event.is_set()
            ):
                raise OllamaStreamCancelled() from exc

            raise OllamaError(
                "Ollama HTTP request failed.",
                code="transport_error",
                retryable=True,
            ) from exc

        finally:
            if response is not None:
                response.close()

            with self._active_stream_lock:
                if self._active_stream_response is response:
                    self._active_stream_response = None

    def cancel_active_stream(self) -> bool:
        """Close the active Ollama HTTP response."""
        with self._active_stream_lock:
            self._cancel_requested = True
            response = self._active_stream_response

        if response is None:
            return False

        try:
            response.close()
        except requests.exceptions.RequestException:
            return False

        return True

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any],
    ) -> Iterator[str]:
        """Stream text; fail rather than silently accept a truncated stream."""
        self._validate_request(messages, options)

        self.last_stream_metrics = None
        self.last_chat_metrics = None

        started = perf_counter()
        first_content_at: float | None = None
        received_final_chunk = False

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": options,
        }

        try:
            with self.session.post(
                self.chat_url,
                json=payload,
                stream=True,
                timeout=(10, self.timeout_seconds),
            ) as response:
                if not response.ok:
                    self._raise_http_error(response)

                for line in response.iter_lines(
                    decode_unicode=True,
                ):
                    if not line:
                        continue

                    if isinstance(line, bytes):
                        line = line.decode(
                            "utf-8",
                            errors="strict",
                        )

                    try:
                        data = json.loads(line)
                    except (json.JSONDecodeError, UnicodeError) as exc:
                        raise OllamaError(
                            "Ollama sent a malformed streaming chunk.",
                            code="invalid_stream_chunk",
                        ) from exc

                    if not isinstance(data, dict):
                        raise OllamaError(
                            "Ollama streaming chunk must be an object.",
                            code="invalid_stream_chunk",
                        )

                    if data.get("error"):
                        raise OllamaError(
                            str(data["error"]),
                            code="model_error",
                        )

                    message = data.get("message")
                    if not isinstance(message, dict):
                        raise OllamaError(
                            "Streaming chunk is missing its message.",
                            code="invalid_stream_chunk",
                        )

                    content = message.get("content", "")
                    done = data.get("done")

                    if not isinstance(content, str):
                        raise OllamaError(
                            "Streaming content must be text.",
                            code="invalid_stream_chunk",
                        )

                    if not isinstance(done, bool):
                        raise OllamaError(
                            "Streaming chunk has no valid completion flag.",
                            code="invalid_stream_chunk",
                        )

                    received_at = perf_counter()

                    if content and first_content_at is None:
                        first_content_at = received_at

                    if done:
                        metrics = {
                            key: data[key]
                            for key in self._METRIC_FIELDS
                            if key in data
                        }
                        metrics.update({
                            "model": data.get("model", self.model),
                            "request_wall_seconds": (
                                received_at - started
                            ),
                            "first_content_seconds": (
                                first_content_at - started
                                if first_content_at is not None
                                else None
                            ),
                            "stream_completed": True,
                        })

                        self.last_stream_metrics = metrics
                        received_final_chunk = True

                    if content:
                        yield content

                    if done:
                        break

            if not received_final_chunk:
                raise OllamaError(
                    "Ollama stream ended before its final completion "
                    "chunk. The response must not be committed as complete.",
                    code="incomplete_stream",
                    retryable=True,
                )

        except OllamaError:
            raise
        except requests.exceptions.Timeout as exc:
            raise OllamaError(
                f"Ollama streaming timed out after "
                f"{self.timeout_seconds:g} seconds.",
                code="timeout",
                retryable=True,
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise OllamaError(
                "Ollama connection was interrupted during streaming.",
                code="connection_error",
                retryable=True,
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaError(
                "Ollama streaming transport failed.",
                code="transport_error",
                retryable=True,
            ) from exc

    def stream_chat_events(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        format: str | dict[str, Any] | None = None,
        cancel_event: Event | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream content and completed tool calls from Ollama."""
        self._validate_request(messages, options)

        if tools is not None and (
            not isinstance(tools, list) or not tools
        ):
            raise OllamaError(
                "Tools must be a non-empty application-owned list.",
                code="invalid_request",
            )

        if format is not None and (
            format != "json" and not isinstance(format, dict)
        ):
            raise OllamaError(
                "Invalid output format.",
                code="invalid_request",
            )

        if cancel_event is not None and cancel_event.is_set():
            raise OllamaStreamCancelled()

        self.last_stream_metrics = None
        self.last_chat_metrics = None

        with self._active_stream_lock:
            self._cancel_requested = False

        started = perf_counter()
        first_content_at: float | None = None
        received_final_chunk = False
        tool_calls: dict[int, dict[str, Any]] = {}

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": options,
        }

        if tools is not None:
            payload["tools"] = tools

        if format is not None:
            payload["format"] = format

        response: requests.Response | None = None

        try:
            response = self.session.post(
                self.chat_url,
                json=payload,
                stream=True,
                timeout=(10, self.timeout_seconds),
            )

            with self._active_stream_lock:
                self._active_stream_response = response
                cancelled_before_response = self._cancel_requested

            if cancelled_before_response or (
                cancel_event is not None and cancel_event.is_set()
            ):
                raise OllamaStreamCancelled()

            if not response.ok:
                self._raise_http_error(response)

            for line in response.iter_lines(decode_unicode=True):
                if self._cancel_requested or (
                    cancel_event is not None and cancel_event.is_set()
                ):
                    raise OllamaStreamCancelled()

                if perf_counter() - started > self.timeout_seconds:
                    raise OllamaError(
                        "Ollama stream exceeded its timeout.",
                        code="timeout",
                        retryable=True,
                    )

                if not line:
                    continue

                if isinstance(line, bytes):
                    if len(line) > 1_048_576:
                        raise OllamaError(
                            "Stream chunk exceeded the size limit.",
                            code="stream_chunk_too_large",
                        )

                    try:
                        line = line.decode("utf-8", errors="strict")
                    except UnicodeDecodeError as exc:
                        raise OllamaError(
                            "Ollama returned invalid UTF-8.",
                            code="invalid_stream_chunk",
                        ) from exc

                elif len(line.encode("utf-8")) > 1_048_576:
                    raise OllamaError(
                        "Stream chunk exceeded the size limit.",
                        code="stream_chunk_too_large",
                    )

                try:
                    data = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_json_pairs,
                    )
                except (ValueError, UnicodeError) as exc:
                    raise OllamaError(
                        "Ollama returned malformed streaming JSON.",
                        code="invalid_stream_chunk",
                    ) from exc

                if not isinstance(data, dict):
                    raise OllamaError(
                        "Streaming chunk must be an object.",
                        code="invalid_stream_chunk",
                    )

                if data.get("error"):
                    raise OllamaError(
                        "Ollama reported a model-side error.",
                        code="model_error",
                    )

                message = data.get("message")

                if not isinstance(message, dict):
                    raise OllamaError(
                        "Streaming chunk is missing its message.",
                        code="invalid_stream_chunk",
                    )

                if message.get("role") not in (None, "assistant"):
                    raise OllamaError(
                        "Unexpected streaming message role.",
                        code="invalid_stream_chunk",
                    )

                content = message.get("content", "")
                done = data.get("done")

                if not isinstance(content, str) or not isinstance(done, bool):
                    raise OllamaError(
                        "Invalid streaming content or completion flag.",
                        code="invalid_stream_chunk",
                    )

                raw_calls = message.get("tool_calls") or []

                if not isinstance(raw_calls, list):
                    raise OllamaError(
                        "Streamed tool_calls must be a list.",
                        code="invalid_tool_call",
                    )

                for index, raw_call in enumerate(raw_calls):
                    self._merge_streamed_tool_call(
                        tool_calls,
                        raw_call,
                        index,
                    )

                now = perf_counter()

                if content:
                    if first_content_at is None:
                        first_content_at = now

                    yield {
                        "type": "content",
                        "content": content,
                    }

                if done:
                    assembled: list[dict[str, Any]] = []

                    for index in sorted(tool_calls):
                        item = tool_calls[index]
                        arguments: Any = item["arguments"]

                        if item["arguments_text"]:
                            if arguments:
                                raise OllamaError(
                                    "Mixed object and text argument fragments.",
                                    code="invalid_tool_call",
                                )

                            arguments = item["arguments_text"]

                        if not item["name"]:
                            raise OllamaError(
                                "Streamed tool call has no function name.",
                                code="invalid_tool_call",
                            )

                        call = {
                            "type": "function",
                            "function": {
                                "name": item["name"],
                                "arguments": arguments,
                            },
                        }

                        if item["id"]:
                            call["id"] = item["id"]

                        assembled.append(call)

                    if len(assembled) > 8:
                        raise OllamaError(
                            "Too many tool calls were returned.",
                            code="invalid_tool_call",
                        )

                    if assembled:
                        yield {
                            "type": "tool_calls",
                            "tool_calls": assembled,
                        }

                    metrics = {
                        key: data[key]
                        for key in self._METRIC_FIELDS
                        if key in data
                    }

                    metrics.update({
                        "model": data.get("model", self.model),
                        "request_wall_seconds": now - started,
                        "first_content_seconds": (
                            first_content_at - started
                            if first_content_at is not None
                            else None
                        ),
                        "stream_completed": True,
                        "tool_call_count": len(assembled),
                    })

                    self.last_stream_metrics = metrics
                    received_final_chunk = True

                    yield {
                        "type": "done",
                        "metrics": metrics,
                    }

                    break

            if not received_final_chunk:
                if self._cancel_requested or (
                    cancel_event is not None and cancel_event.is_set()
                ):
                    raise OllamaStreamCancelled()

                raise OllamaError(
                    "Ollama stream ended before its final completion chunk.",
                    code="incomplete_stream",
                    retryable=True,
                )

        except OllamaError:
            raise

        except requests.exceptions.Timeout as exc:
            if cancel_event is not None and cancel_event.is_set():
                raise OllamaStreamCancelled() from exc

            raise OllamaError(
                "Ollama streaming timed out.",
                code="timeout",
                retryable=True,
            ) from exc

        except requests.exceptions.RequestException as exc:
            with self._active_stream_lock:
                cancelled = self._cancel_requested

            if cancelled or (
                cancel_event is not None and cancel_event.is_set()
            ):
                raise OllamaStreamCancelled() from exc

            raise OllamaError(
                "Ollama streaming transport failed.",
                code="transport_error",
                retryable=True,
            ) from exc

        finally:
            if response is not None:
                response.close()

            with self._active_stream_lock:
                if self._active_stream_response is response:
                    self._active_stream_response = None
        
    @staticmethod
    def _merge_streamed_tool_call(
        calls: dict[int, dict[str, Any]],
        raw_call: Any,
        fallback_index: int,
    ) -> None:
        if not isinstance(raw_call, dict):
            raise OllamaError(
                "Malformed streamed tool call.",
                code="invalid_tool_call",
            )

        raw_index = raw_call.get("index", fallback_index)

        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raw_index = fallback_index

        if not 0 <= raw_index < 16:
            raise OllamaError(
                "Invalid streamed tool-call index.",
                code="invalid_tool_call",
            )

        accumulated = calls.setdefault(
            raw_index,
            {
                "id": None,
                "name": "",
                "arguments": {},
                "arguments_text": "",
            },
        )

        call_id = raw_call.get("id")

        if isinstance(call_id, str) and call_id:
            accumulated["id"] = call_id[:128]

        function = raw_call.get("function")

        if not isinstance(function, dict):
            raise OllamaError(
                "Tool call is missing its function object.",
                code="invalid_tool_call",
            )

        name_part = function.get("name")

        if name_part is not None:
            if not isinstance(name_part, str):
                raise OllamaError(
                    "Tool name fragment must be text.",
                    code="invalid_tool_call",
                )

            current = accumulated["name"]

            if not current:
                accumulated["name"] = name_part
            elif name_part == current or current.endswith(name_part):
                pass
            elif name_part.startswith(current):
                accumulated["name"] = name_part
            else:
                accumulated["name"] = current + name_part

            if len(accumulated["name"]) > 64:
                raise OllamaError(
                    "Tool name is too long.",
                    code="invalid_tool_call",
                )

        arguments = function.get("arguments")

        if arguments is None:
            return

        if isinstance(arguments, dict):
            current = accumulated["arguments"]

            if not isinstance(current, dict):
                raise OllamaError(
                    "Incompatible tool argument fragments.",
                    code="invalid_tool_call",
                )

            for key, value in arguments.items():
                if key in current and current[key] != value:
                    raise OllamaError(
                        "Conflicting streamed tool arguments.",
                        code="invalid_tool_call",
                    )

                current[key] = value

        elif isinstance(arguments, str):
            current = accumulated["arguments_text"]

            if not current:
                accumulated["arguments_text"] = arguments
            elif arguments.startswith(current):
                accumulated["arguments_text"] = arguments
            elif current.endswith(arguments):
                pass
            else:
                accumulated["arguments_text"] = current + arguments

            if len(accumulated["arguments_text"]) > 8192:
                raise OllamaError(
                    "Tool arguments exceeded the size limit.",
                    code="tool_arguments_too_large",
                )

        else:
            raise OllamaError(
                "Tool arguments must be JSON text or an object.",
                code="invalid_tool_call",
            )

    @classmethod
    def _raise_http_error(
        cls,
        response: requests.Response,
    ) -> None:
        status = response.status_code
        detail = cls._http_error_detail(response)

        raise OllamaError(
            f"Ollama returned HTTP {status}: {detail}",
            code="http_error",
            retryable=status in cls._RETRYABLE_HTTP_STATUS,
            status_code=status,
        )

    @staticmethod
    def _http_error_detail(
        response: requests.Response,
    ) -> str:
        try:
            data = response.json()
            if isinstance(data, dict) and data.get("error"):
                return str(data["error"])[:500]
        except (ValueError, requests.exceptions.RequestException):
            pass

        return (response.text or "").strip()[:500] or (
            "No details returned by Ollama."
        )