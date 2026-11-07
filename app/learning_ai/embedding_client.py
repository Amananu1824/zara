"""Ollama embedding client for Zara's semantic retrieval."""
from __future__ import annotations

import json
import math
import re
from typing import Sequence
from threading import Event, Lock, local

import requests

from app.core.config import settings


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails or returns invalid data."""

class EmbeddingCancelled(EmbeddingError):
    """An active embedding request was cancelled."""


class EmbeddingClient:
    """Generate and validate vectors through Ollama's /api/embed endpoint."""

    FORMAT_VERSION = "embeddinggemma_retrieval_v1"

    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.zara_embedding_model
        self._configured_model_version = (
            settings.zara_embedding_model_version.strip()
        )
        self._model_version_cache: str | None = None
        self._model_version_lock = Lock()
        self.dimensions = settings.zara_embedding_dimensions
        self.format_version = self.FORMAT_VERSION
        self.timeout = settings.ollama_timeout_seconds
        self._session_local = local()
        self._active_lock = Lock()
        self._active_responses: dict[int, requests.Response] = {}
        
    @property
    def session(self) -> requests.Session:
        session = getattr(self._session_local, "session", None)

        if session is None:
            session = requests.Session()
            self._session_local.session = session

        return session

    @staticmethod
    def _normalize_model_name(value: str) -> str:
        """
        Normalize Ollama model references.

        A tagless name is equivalent to its :latest tag.
        Namespace/model names are preserved.
        """
        normalized = str(value or "").strip()

        if not normalized:
            return ""

        final_component = normalized.rsplit("/", 1)[-1]

        if ":" not in final_component:
            normalized = f"{normalized}:latest"

        return normalized.casefold()

    @property
    def model_version(self) -> str:
        """
        Resolve an immutable embedding-model identity from Ollama.

        The digest is cached for this client lifetime. Restart Zara after
        replacing/updating the local embedding model.
        """
        with self._model_version_lock:
            if self._model_version_cache:
                return self._model_version_cache

            response: requests.Response | None = None

            try:
                response = self.session.get(
                    f"{self.base_url}/api/tags",
                    timeout=(
                        5.0,
                        max(5.0, min(float(self.timeout), 15.0)),
                    ),
                )
                response.raise_for_status()
                payload = response.json()

            except (
                requests.exceptions.RequestException,
                ValueError,
            ) as exc:
                raise EmbeddingError(
                    "Could not resolve the local Ollama embedding "
                    "model digest."
                ) from exc

            finally:
                if response is not None:
                    response.close()

            models = (
                payload.get("models")
                if isinstance(payload, dict)
                else None
            )

            if not isinstance(models, list):
                raise EmbeddingError(
                    "Ollama model-list response has an invalid structure."
                )

            expected_name = self._normalize_model_name(self.model)
            matched_model: dict[str, object] | None = None

            for model_row in models:
                if not isinstance(model_row, dict):
                    continue

                candidates = (
                    model_row.get("name"),
                    model_row.get("model"),
                )

                if any(
                    isinstance(candidate, str)
                    and self._normalize_model_name(candidate)
                    == expected_name
                    for candidate in candidates
                ):
                    matched_model = model_row
                    break

            if matched_model is None:
                raise EmbeddingError(
                    "Configured embedding model is not installed "
                    "in the connected Ollama instance."
                )

            digest = str(
                matched_model.get("digest") or ""
            ).strip().casefold()

            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise EmbeddingError(
                    "Ollama returned a missing or invalid model digest."
                )

            resolved_version = f"sha256:{digest}"

            configured_version = (
                self._configured_model_version.strip().casefold()
            )

            if configured_version not in {
                "",
                "unverified",
                resolved_version,
            }:
                raise EmbeddingError(
                    "Configured embedding model version does not match "
                    "the installed Ollama model digest."
                )

            self._model_version_cache = resolved_version
            return resolved_version
        
    @staticmethod
    def format_document(title: str, content: str) -> str:
        """Format a document using EmbeddingGemma's retrieval convention."""
        clean_title = (title or "").strip() or "none"
        clean_content = (content or "").strip()

        if not clean_content:
            raise ValueError("Cannot embed an empty document.")

        return f"title: {clean_title} | text: {clean_content}"

    @staticmethod
    def format_query(query: str) -> str:
        """Format a retrieval query using EmbeddingGemma's query prefix."""
        clean_query = (query or "").strip()

        if not clean_query:
            raise ValueError("Cannot embed an empty query.")

        return f"task: search result | query: {clean_query}"

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        cancel_event: Event | None = None,
    ) -> list[list[float]]:
        """Generate bounded embeddings with request-scoped cancellation."""
        clean_texts = [text.strip() for text in texts]

        if not clean_texts:
            return []

        if any(not text for text in clean_texts):
            raise ValueError("Embedding input contains empty text.")

        if cancel_event is not None and cancel_event.is_set():
            raise EmbeddingCancelled("Embedding request cancelled.")

        response: requests.Response | None = None
        response_key = (
            id(cancel_event)
            if cancel_event is not None
            else None
        )

        try:
            response = self.session.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": clean_texts,
                },
                timeout=(5.0, self.timeout),
                stream=True,
            )

            if response_key is not None:
                with self._active_lock:
                    if cancel_event.is_set():
                        raise EmbeddingCancelled(
                            "Embedding request cancelled."
                        )

                    self._active_responses[response_key] = response

            body = bytearray()

            for chunk in response.iter_content(
                chunk_size=64 * 1024
            ):
                if cancel_event is not None and cancel_event.is_set():
                    raise EmbeddingCancelled(
                        "Embedding request cancelled."
                    )

                if not chunk:
                    continue

                body.extend(chunk)

                if len(body) > 8 * 1024 * 1024:
                    raise EmbeddingError(
                        "Embedding response exceeded the size limit."
                    )

            if cancel_event is not None and cancel_event.is_set():
                raise EmbeddingCancelled(
                    "Embedding request cancelled."
                )

            try:
                data = json.loads(body.decode("utf-8-sig"))
            except (ValueError, UnicodeError) as exc:
                raise EmbeddingError(
                    "Ollama returned invalid embedding JSON."
                ) from exc

            if not response.ok:
                detail = (
                    data.get("error")
                    if isinstance(data, dict)
                    else None
                )

                raise EmbeddingError(
                    f"Ollama embedding HTTP {response.status_code}: "
                    f"{detail or 'request failed'}"
                )

            if isinstance(data, dict) and data.get("error"):
                raise EmbeddingError(str(data["error"]))

            vectors = (
                data.get("embeddings")
                if isinstance(data, dict)
                else None
            )

            if (
                not isinstance(vectors, list)
                or len(vectors) != len(clean_texts)
            ):
                raise EmbeddingError(
                    "Ollama returned an unexpected vector count."
                )

            result: list[list[float]] = []

            for index, vector in enumerate(vectors):
                if not isinstance(vector, list):
                    raise EmbeddingError(
                        f"Vector {index} is not a list."
                    )

                if len(vector) != self.dimensions:
                    raise EmbeddingError(
                        f"Vector {index} has {len(vector)} dimensions; "
                        f"expected {self.dimensions}."
                    )

                try:
                    values = [float(value) for value in vector]
                except (TypeError, ValueError) as exc:
                    raise EmbeddingError(
                        f"Vector {index} contains invalid values."
                    ) from exc

                if not all(math.isfinite(value) for value in values):
                    raise EmbeddingError(
                        f"Vector {index} contains non-finite values."
                    )

                result.append(values)

            return result

        except EmbeddingCancelled:
            raise

        except requests.exceptions.Timeout as exc:
            if cancel_event is not None and cancel_event.is_set():
                raise EmbeddingCancelled(
                    "Embedding request cancelled."
                ) from exc

            raise EmbeddingError(
                f"Embedding request timed out after {self.timeout:g} seconds."
            ) from exc

        except requests.exceptions.ConnectionError as exc:
            if cancel_event is not None and cancel_event.is_set():
                raise EmbeddingCancelled(
                    "Embedding request cancelled."
                ) from exc

            raise EmbeddingError(
                "Could not connect to Ollama embedding service."
            ) from exc

        except requests.exceptions.RequestException as exc:
            if cancel_event is not None and cancel_event.is_set():
                raise EmbeddingCancelled(
                    "Embedding request cancelled."
                ) from exc

            raise EmbeddingError(
                "Ollama embedding transport failed."
            ) from exc

        finally:
            if response is not None:
                if response_key is not None:
                    with self._active_lock:
                        if (
                            self._active_responses.get(response_key)
                            is response
                        ):
                            self._active_responses.pop(response_key, None)

                response.close()

    def cancel_active_request(
        self,
        cancel_event: Event | None,
    ) -> bool:
        """Best-effort interruption of one request's active HTTP response."""
        if cancel_event is None:
            return False

        with self._active_lock:
            response = self._active_responses.get(id(cancel_event))

        if response is None:
            return False

        try:
            response.close()
            return True
        except requests.exceptions.RequestException:
            return False

    def embed_query(
        self,
        query: str,
        *,
        cancel_event: Event | None = None,
    ) -> list[float]:
        formatted = self.format_query(query)

        return self.embed_texts(
            [formatted],
            cancel_event=cancel_event,
        )[0]

    def embed_documents(
        self,
        documents: Sequence[tuple[str, str]],
        *,
        cancel_event: Event | None = None,
    ) -> list[list[float]]:
        formatted = [
            self.format_document(title, content)
            for title, content in documents
        ]

        return self.embed_texts(
            formatted,
            cancel_event=cancel_event,
        )