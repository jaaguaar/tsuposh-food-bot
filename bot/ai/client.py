from __future__ import annotations

import json
import logging
from typing import Any

from bot.config import settings

logger = logging.getLogger(__name__)


class AiClientError(Exception):
    pass


class AzureAiInferenceClient:
    """Compatibility wrapper for Azure OpenAI / Foundry v1 endpoint."""

    def __init__(self) -> None:
        self._base_url = (settings.resolved_ai_base_url or "").strip()
        self._api_key = (settings.resolved_ai_api_key or "").strip()
        self._deployment = (settings.resolved_ai_deployment or "").strip()
        self._enabled = bool(self._base_url and self._api_key and self._deployment)
        self._client = None
        self._aclient = None

        if not self._enabled:
            logger.warning("Azure OpenAI client is disabled: missing base_url/key/deployment")
            return

        try:
            from openai import AsyncOpenAI, OpenAI
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
            self._aclient = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)
        except Exception as ex:
            logger.exception("Failed to initialize Azure OpenAI client")
            raise AiClientError(str(ex)) from ex

    @property
    def enabled(self) -> bool:
        return self._enabled and self._aclient is not None

    async def acomplete_text(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
        if not self.enabled:
            raise AiClientError("Azure OpenAI client is not configured")
        assert self._aclient is not None
        try:
            response = await self._aclient.chat.completions.create(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max_tokens,
            )
            return self._extract_text(response)
        except Exception as ex:
            logger.exception("Azure OpenAI async text completion failed")
            raise AiClientError(str(ex)) from ex

    async def acomplete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> dict[str, Any]:
        _ = temperature  # gpt-5.x Azure chat may reject non-default temperature
        content = await self.acomplete_text(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens)
        return self._parse_json(content)

    # Backward-compatible sync wrappers
    def complete_text(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
        if self._client is None:
            raise AiClientError("Azure OpenAI client is not configured")
        try:
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max_tokens,
            )
            return self._extract_text(response)
        except Exception as ex:
            logger.exception("Azure OpenAI text completion failed")
            raise AiClientError(str(ex)) from ex

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> dict[str, Any]:
        _ = temperature
        return self._parse_json(self.complete_text(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens))

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            choices = getattr(response, "choices", None)
            if choices:
                message = getattr(choices[0], "message", None)
                if message is not None:
                    content = getattr(message, "content", None)
                    if isinstance(content, str) and content.strip():
                        return content
                    if isinstance(content, list):
                        parts: list[str] = []
                        for item in content:
                            if isinstance(item, str):
                                if item.strip():
                                    parts.append(item)
                            else:
                                text_value = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                                if isinstance(text_value, str) and text_value.strip():
                                    parts.append(text_value)
                        if parts:
                            return "\n".join(parts)

            as_dict = response.model_dump() if hasattr(response, "model_dump") else None
            if isinstance(as_dict, dict):
                choices = as_dict.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content

            raise AiClientError("Could not extract text content from model response")
        except Exception as ex:
            raise AiClientError(f"Failed to extract response text: {ex}") from ex

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        raw = text.strip()

        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        candidates: list[str] = []
        seen: set[str] = set()

        def _add_candidate(value: str) -> None:
            v = value.strip()
            if v and v not in seen:
                seen.add(v)
                candidates.append(v)

        _add_candidate(raw)

        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            _add_candidate(raw[start : end + 1])

        if start != -1:
            _add_candidate(raw[start:])

        def _try_parse(candidate: str) -> dict[str, Any] | None:
            attempts = [candidate]
            if '""' in candidate:
                attempts.append(candidate.replace('""', '"'))

            for attempt in attempts:
                try:
                    data = json.loads(attempt)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass

                open_braces = attempt.count("{")
                close_braces = attempt.count("}")
                if open_braces > close_braces:
                    fixed = attempt + ("}" * (open_braces - close_braces))
                    try:
                        data = json.loads(fixed)
                        if isinstance(data, dict):
                            return data
                    except json.JSONDecodeError:
                        pass
            return None

        for candidate in candidates:
            parsed = _try_parse(candidate)
            if parsed is not None:
                return parsed

        preview = raw[:800].replace("\n", "\\n")
        raise AiClientError(f"Invalid JSON from model. Raw preview: {preview}")
