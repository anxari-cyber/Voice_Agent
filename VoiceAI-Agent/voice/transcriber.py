from __future__ import annotations

from voice.cuda_runtime import configure_cuda_dlls

configure_cuda_dlls()

from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_name: str, device: str = "auto", compute_type: str = "auto") -> None:
        self.model_name = model_name
        requested_device = device
        if device == "auto":
            device = "cuda"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        try:
            self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
            self.device = device
        except RuntimeError as error:
            if requested_device != "auto" or device != "cuda":
                raise
            print(f"CUDA unavailable ({error}). Falling back to CPU int8.")
            self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
            self.device = "cpu"

    def transcribe(self, audio: object) -> str:
        try:
            return self._transcribe(audio)
        except RuntimeError as error:
            if self.device != "cuda" or not _is_cuda_error(error):
                raise
            print(f"CUDA inference unavailable ({error}). Falling back to CPU int8.")
            self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            self.device = "cpu"
            return self._transcribe(audio)

    def _transcribe(self, audio: object) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-0.6,
            no_speech_threshold=0.4,
            vad_filter=True,
            vad_parameters={
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 500,
            },
        )
        # faster-whisper already skips a segment when no_speech_prob > no_speech_threshold AND
        # avg_logprob < log_prob_threshold. A second, stricter OR filter here used to drop
        # short or quiet commands that were transcribed correctly.
        texts = (segment.text.strip() for segment in segments)
        return " ".join(text for text in texts if text).strip()


def _is_cuda_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in ("cuda", "cublas", "cudnn", "gpu"))
