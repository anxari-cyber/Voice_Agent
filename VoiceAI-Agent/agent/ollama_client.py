from __future__ import annotations

import httpx


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        num_predict: int = 128,
        num_ctx: int = 2048,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_predict = num_predict
        self.num_ctx = num_ctx

    def generate(self, prompt: str) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "keep_alive": "10m",
                    "options": {
                        "num_predict": self.num_predict,
                        "num_ctx": self.num_ctx,
                        "temperature": 0,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise OllamaError(f"Ollama request failed: {error}") from error

        result = response.json()
        response_text = result.get("response")
        if not isinstance(response_text, str):
            raise OllamaError("Ollama returned an invalid response.")
        return response_text.strip()
