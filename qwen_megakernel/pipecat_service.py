"""Pipecat TTSService adapter for the Qwen3-TTS megakernel backend."""

from __future__ import annotations

from typing import AsyncGenerator

from qwen_megakernel.tts_stream import MegakernelQwen3TTS, TTSMetrics


class QwenMegakernelTTSService:
    """Lazy Pipecat service wrapper.

    Pipecat is an optional dependency for this repo, so imports happen inside
    ``__init__``. The class subclasses Pipecat's TTSService at runtime by
    composition-compatible inheritance through ``_Base``.
    """

    def __new__(cls, *args, **kwargs):
        from pipecat.services.tts_service import TextAggregationMode, TTSService

        class _Service(TTSService):
            def __init__(
                self,
                *,
                engine: MegakernelQwen3TTS,
                ref_audio,
                ref_text: str | None = None,
                language: str = "English",
                x_vector_only_mode: bool = True,
                code_chunk_steps: int = 6,
                log_path: str | None = None,
                **service_kwargs,
            ):
                from qwen_megakernel.tts_stream import TTSMetrics

                service_kwargs.setdefault("sample_rate", 24000)
                service_kwargs.setdefault("text_aggregation_mode", TextAggregationMode.SENTENCE)
                service_kwargs.setdefault("push_start_frame", True)
                super().__init__(**service_kwargs)
                self.engine = engine
                self.ref_audio = ref_audio
                self.ref_text = ref_text
                self.language = language
                self.x_vector_only_mode = x_vector_only_mode
                self.code_chunk_steps = code_chunk_steps
                self.log_path = log_path
                self.last_metrics = TTSMetrics()

            async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[object, None]:
                from pipecat.frames.frames import TTSAudioRawFrame

                self.last_metrics = TTSMetrics()
                async for pcm, sr in self.engine.stream_pcm_chunks(
                    text,
                    language=self.language,
                    ref_audio=self.ref_audio,
                    ref_text=self.ref_text,
                    x_vector_only_mode=self.x_vector_only_mode,
                    code_chunk_steps=self.code_chunk_steps,
                    metrics=self.last_metrics,
                    log_path=self.log_path,
                ):
                    yield TTSAudioRawFrame(
                        audio=pcm,
                        sample_rate=sr,
                        num_channels=1,
                        context_id=context_id,
                    )

        return _Service(*args, **kwargs)
