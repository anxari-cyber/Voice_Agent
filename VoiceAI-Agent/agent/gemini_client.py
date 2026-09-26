from __future__ import annotations

from google import genai
from google.genai import types


class GeminiError(RuntimeError):
    pass


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.8-flash",
        timeout_ms: int = 30_000,
    ) -> None:
        if not api_key.strip():
            raise GeminiError("GEMINI_API_KEY is not configured.")
        self.model = model
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=128,
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
        except Exception as error:
            raise GeminiError(f"Gemini request failed: {error}") from error

        response_text = getattr(response, "text", None)
        if not isinstance(response_text, str):
            response_text = getattr(response, "output_text", None)
        if not isinstance(response_text, str) or not response_text.strip():
            raise GeminiError("Gemini returned an empty response.")
        return response_text.strip()