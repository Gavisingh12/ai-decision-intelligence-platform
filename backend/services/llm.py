"""LLM provider abstraction for OpenAI and Gemini."""

from __future__ import annotations

from dataclasses import dataclass

from backend.core.config import get_settings


@dataclass
class LLMResult:
    """Normalized text generation result."""

    answer: str
    provider: str


class LLMService:
    """Minimal provider wrapper with a graceful local fallback."""

    def generate(self, *, prompt: str, system_prompt: str, provider: str | None = None) -> LLMResult:
        """Generate a response using the configured provider."""

        settings = get_settings()
        resolved_provider = (provider or settings.default_llm_provider).lower()

        if resolved_provider == "openai" and settings.openai_api_key:
            return LLMResult(
                answer=self._generate_openai(prompt=prompt, system_prompt=system_prompt),
                provider="openai",
            )

        if resolved_provider == "gemini" and settings.gemini_api_key:
            return LLMResult(
                answer=self._generate_gemini(prompt=prompt, system_prompt=system_prompt),
                provider="gemini",
            )

        if settings.openai_api_key:
            return LLMResult(
                answer=self._generate_openai(prompt=prompt, system_prompt=system_prompt),
                provider="openai",
            )

        if settings.gemini_api_key:
            return LLMResult(
                answer=self._generate_gemini(prompt=prompt, system_prompt=system_prompt),
                provider="gemini",
            )

        fallback = (
            "External LLM keys are not configured, so this is a grounded local fallback.\n\n"
            f"{prompt[:1800]}"
        )
        return LLMResult(answer=fallback, provider="fallback")

    def _generate_openai(self, *, prompt: str, system_prompt: str) -> str:
        """Call the OpenAI Chat Completions API."""

        from openai import OpenAI

        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def _generate_gemini(self, *, prompt: str, system_prompt: str) -> str:
        """Call the Gemini API."""

        import google.generativeai as genai

        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)
        response = model.generate_content(f"{system_prompt}\n\n{prompt}")
        return (response.text or "").strip()


llm_service = LLMService()
