"""Streaming Qwen3-TTS pipeline using the megakernel for the talker decoder."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import torch

from qwen_megakernel.tts_kernel import CodePredictorMegakernel, TTSTalkerMegakernel


@dataclass
class TTSMetrics:
    ttfc_ms: float | None = None
    first_codec_ms: float | None = None
    prompt_prepare_ms: float | None = None
    prefill_ms: float | None = None
    generation_ms: float | None = None
    vocoder_ms: float = 0.0
    megakernel_step_ms: float = 0.0
    code_predictor_ms: float = 0.0
    total_ms: float | None = None
    rtf: float | None = None
    codec_tokens_per_s: float | None = None
    generated_audio_s: float = 0.0
    decode_s: float = 0.0
    chunks: int = 0
    codec_steps: int = 0
    notes: list[str] = field(default_factory=list)


class TimingLogger:
    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else None
        self._t0 = time.perf_counter()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def event(self, name: str, **fields) -> None:
        if not self.path:
            return
        row = {
            "event": name,
            "t_ms": (time.perf_counter() - self._t0) * 1000.0,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def float_wav_to_pcm16(wav) -> bytes:
    if hasattr(wav, "cpu"):
        wav = wav.cpu()
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype("<i2").tobytes()


class MegakernelQwen3TTS:
    """High-level voice-clone streaming facade.

    The official Qwen3-TTS package still owns text/audio preprocessing, speaker
    encoding, the codebook predictor, and waveform decoding. The megakernel
    replaces the 28-layer talker transformer forward pass.
    """

    def __init__(self, qwen_tts_model, max_seq_len: int | None = None):
        self.wrapper = qwen_tts_model
        self.model = qwen_tts_model.model
        self.talker = self.model.talker
        self.kernel = TTSTalkerMegakernel(self.talker, max_seq_len=max_seq_len)
        self.code_predictor_kernel = CodePredictorMegakernel(self.talker.code_predictor)
        self._voice_prompt_cache: dict[tuple[object, ...], tuple[dict, list, torch.Tensor | None]] = {}
        # Serializes requests: the talker KV cache is stateful; concurrent calls corrupt it.
        self._inference_lock = asyncio.Lock()

        # Pre-stack codec embedding tables for O(1) batch lookup.
        # Replaces 15 separate Python→CUDA dispatch calls per step (~57 ms) with one
        # fancy-index op on a [15, vocab, hidden] tensor (~1 ms).
        _raw_embeds = list(self.talker.code_predictor.get_input_embeddings())
        self._codec_embeds_stacked = torch.stack(
            [e.weight.to(dtype=torch.bfloat16, device="cuda").contiguous() for e in _raw_embeds],
            dim=0,
        )  # [num_codebook_groups-1, vocab_size, hidden_size]
        self._codec_embed_idx = torch.arange(len(_raw_embeds), dtype=torch.long, device="cuda")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        *,
        max_seq_len: int | None = None,
        **kwargs,
    ) -> "MegakernelQwen3TTS":
        from qwen_tts import Qwen3TTSModel

        kwargs.setdefault("device_map", "cuda:0")
        kwargs.setdefault("dtype", torch.bfloat16)
        qwen_tts_model = Qwen3TTSModel.from_pretrained(model_name, **kwargs)
        return cls(qwen_tts_model, max_seq_len=max_seq_len)

    @staticmethod
    def _validate_ref_audio(ref_audio) -> None:
        if isinstance(ref_audio, (str, os.PathLike)) and not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    def _voice_cache_key(
        self,
        ref_audio,
        ref_text: str | None,
        x_vector_only_mode: bool,
    ) -> tuple[object, ...]:
        if isinstance(ref_audio, str) and os.path.exists(ref_audio):
            stat = os.stat(ref_audio)
            audio_key: tuple[object, ...] = ("path", os.path.abspath(ref_audio), stat.st_mtime_ns, stat.st_size)
        else:
            audio_key = ("value", str(ref_audio))
        return (*audio_key, ref_text or "", bool(x_vector_only_mode))

    def _get_cached_voice_prompt(
        self,
        *,
        ref_audio,
        ref_text: str | None,
        x_vector_only_mode: bool,
    ) -> tuple[dict, list, torch.Tensor | None]:
        key = self._voice_cache_key(ref_audio, ref_text, x_vector_only_mode)
        cached = self._voice_prompt_cache.get(key)
        if cached is not None:
            return cached

        prompt_items = self.wrapper.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )
        voice_clone_prompt = self.wrapper._prompt_items_to_voice_clone_prompt(prompt_items)
        voice_clone_spk_embeds = self.model.generate_speaker_prompt(voice_clone_prompt)
        ref_id = None
        if ref_text:
            ref_id = self.wrapper._tokenize_texts([self.wrapper._build_ref_text(ref_text)])[0]

        cached = (voice_clone_prompt, voice_clone_spk_embeds, ref_id)
        self._voice_prompt_cache[key] = cached
        return cached

    def clear_voice_prompt_cache(self) -> None:
        self._voice_prompt_cache.clear()

    def warm_speaker(
        self,
        ref_audio,
        ref_text: str | None = None,
        x_vector_only_mode: bool = True,
    ) -> None:
        """Pre-encode a reference speaker so the first real request has zero cold-start overhead.

        Call this once after loading the model, before serving requests.
        """
        self._validate_ref_audio(ref_audio)
        self._get_cached_voice_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )

    def _prepare_voice_clone_inputs(
        self,
        text: str,
        *,
        language: str,
        ref_audio,
        ref_text: str | None,
        x_vector_only_mode: bool,
        non_streaming_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        prepare_t0 = time.perf_counter()
        voice_clone_prompt, voice_clone_spk_embeds, ref_id = self._get_cached_voice_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )
        input_id = self.wrapper._tokenize_texts([self.wrapper._build_assistant_text(text)])[0]

        speaker_embed = (
            voice_clone_spk_embeds[0]
            if voice_clone_prompt["x_vector_only_mode"][0] or voice_clone_prompt["icl_mode"][0]
            else None
        )

        if language.lower() == "auto":
            language_id = None
        else:
            language_id = self.model.config.talker_config.codec_language_id[language.lower()]

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(
                torch.tensor(
                    [[self.model.config.tts_bos_token_id, self.model.config.tts_eos_token_id, self.model.config.tts_pad_token_id]],
                    device=self.talker.device,
                    dtype=input_id.dtype,
                )
            )
        ).chunk(3, dim=1)

        if language_id is None:
            codec_prefill_list = [[
                self.model.config.talker_config.codec_nothink_id,
                self.model.config.talker_config.codec_think_bos_id,
                self.model.config.talker_config.codec_think_eos_id,
            ]]
        else:
            codec_prefill_list = [[
                self.model.config.talker_config.codec_think_id,
                self.model.config.talker_config.codec_think_bos_id,
                language_id,
                self.model.config.talker_config.codec_think_eos_id,
            ]]

        codec_embed_0 = self.talker.get_input_embeddings()(
            torch.tensor(codec_prefill_list, device=self.talker.device, dtype=input_id.dtype)
        )
        codec_embed_1 = self.talker.get_input_embeddings()(
            torch.tensor(
                [[self.model.config.talker_config.codec_pad_id, self.model.config.talker_config.codec_bos_id]],
                device=self.talker.device,
                dtype=input_id.dtype,
            )
        )
        if speaker_embed is None:
            codec_embed = torch.cat([codec_embed_0, codec_embed_1], dim=1)
        else:
            codec_embed = torch.cat([codec_embed_0, speaker_embed.view(1, 1, -1), codec_embed_1], dim=1)

        role_embed = self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, :3]))
        head_embed = torch.cat(
            (tts_pad_embed.expand(-1, codec_embed.shape[1] - 2, -1), tts_bos_embed),
            dim=1,
        ) + codec_embed[:, :-1]
        talker_input_embed = torch.cat((role_embed, head_embed), dim=1)

        if voice_clone_prompt["ref_code"] is not None and voice_clone_prompt["ref_code"][0] is not None and voice_clone_prompt["icl_mode"][0]:
            icl_input_embed, trailing_text_hidden = self.model.generate_icl_prompt(
                text_id=input_id[:, 3:-5],
                ref_id=ref_id[:, 3:-2],
                ref_code=voice_clone_prompt["ref_code"][0].to(self.talker.device),
                tts_pad_embed=tts_pad_embed,
                tts_eos_embed=tts_eos_embed,
                non_streaming_mode=non_streaming_mode,
            )
            talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
        else:
            talker_input_embed = torch.cat(
                [
                    talker_input_embed,
                    self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 3:4]))
                    + codec_embed[:, -1:],
                ],
                dim=1,
            )
            trailing_text_hidden = torch.cat(
                (
                    self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 4:-5])),
                    tts_eos_embed,
                ),
                dim=1,
            )

        return (
            talker_input_embed.to(torch.bfloat16).contiguous(),
            trailing_text_hidden.to(torch.bfloat16).contiguous(),
            tts_pad_embed.to(torch.bfloat16).contiguous(),
            (time.perf_counter() - prepare_t0) * 1000.0,
        )

    @torch.inference_mode()
    async def stream_codebooks(
        self,
        text: str,
        *,
        language: str = "English",
        ref_audio=None,
        ref_text: str | None = None,
        x_vector_only_mode: bool = True,
        max_new_tokens: int = 4096,
        max_repeat: int = 4,
        generation_timeout_s: float = 30.0,
        metrics: TTSMetrics | None = None,
        timing_logger: TimingLogger | None = None,
    ) -> AsyncIterator[torch.Tensor]:
        if ref_audio is None:
            raise ValueError("ref_audio is required for the Base voice-clone model.")
        self._validate_ref_audio(ref_audio)

        async with self._inference_lock:
            t0 = time.perf_counter()
            deadline = t0 + generation_timeout_s

            try:
                prompt_embeds, trailing_text, tts_pad_embed, prompt_prepare_ms = self._prepare_voice_clone_inputs(
                    text,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                )
                if metrics is not None:
                    metrics.prompt_prepare_ms = prompt_prepare_ms
                if timing_logger is not None:
                    timing_logger.event("prompt_prepared", prompt_prepare_ms=prompt_prepare_ms, prompt_tokens=prompt_embeds.shape[1])

                self.kernel.reset()
                next_first_code = None
                prefill_t0 = time.perf_counter()
                n_prefill = prompt_embeds.shape[1]
                for i in range(n_prefill - 1):
                    self.kernel.step_embed_async(prompt_embeds[:, i : i + 1, :])
                # Final prefill token: one sync to get first_code for EOS check
                step_t0 = time.perf_counter()
                next_first_code = self.kernel.step_embed(prompt_embeds[:, -1:, :])
                if metrics is not None:
                    metrics.megakernel_step_ms += (time.perf_counter() - step_t0) * 1000.0
                past_hidden = self.kernel.last_hidden_bf16
                if metrics is not None:
                    metrics.prefill_ms = (time.perf_counter() - prefill_t0) * 1000.0
                if timing_logger is not None:
                    timing_logger.event("prefill_done", prefill_ms=(time.perf_counter() - prefill_t0) * 1000.0, prompt_tokens=prompt_embeds.shape[1])

                codebook_count = 0
                eos = self.model.config.talker_config.codec_eos_token_id
                generation_t0 = time.perf_counter()
                _prev_code: int | None = None
                _repeat_count: int = 0
                for generation_step in range(max_new_tokens):
                    if time.perf_counter() > deadline:
                        raise asyncio.TimeoutError(
                            f"TTS generation timed out after {generation_timeout_s}s at step {generation_step}"
                        )

                    if next_first_code == eos:
                        if timing_logger is not None:
                            timing_logger.event("eos", generation_step=generation_step)
                        break
                    # Stop if the same token repeats: megakernel can diverge from the reference
                    # and get stuck in a loop. max_repeat consecutive identical tokens = done.
                    if next_first_code == _prev_code:
                        _repeat_count += 1
                        if _repeat_count >= max_repeat:
                            if timing_logger is not None:
                                timing_logger.event("repeat_stop", generation_step=generation_step, code=next_first_code)
                            break
                    else:
                        _prev_code = next_first_code
                        _repeat_count = 1

                    input_ids = torch.tensor([[next_first_code]], dtype=torch.long, device=self.talker.device)
                    last_id_hidden = self.talker.get_input_embeddings()(input_ids).to(torch.bfloat16)
                    predictor_t0 = time.perf_counter()
                    mk_sequences = self.code_predictor_kernel.generate(past_hidden, last_id_hidden)
                    if metrics is not None:
                        metrics.code_predictor_ms += (time.perf_counter() - predictor_t0) * 1000.0
                    codec_ids = torch.cat((input_ids, mk_sequences), dim=-1)
                    yield codec_ids[0].detach()
                    codebook_count += 1
                    if metrics is not None:
                        metrics.codec_steps = codebook_count
                    if timing_logger is not None:
                        timing_logger.event("codebook", generation_step=generation_step, first_code=int(next_first_code))

                    # Batch lookup: one CUDA index op instead of 15 Python→CUDA dispatch calls.
                    # _codec_embeds_stacked: [15, vocab, hidden], mk_sequences[0]: [15]
                    pred_embeds = self._codec_embeds_stacked[self._codec_embed_idx, mk_sequences[0]]
                    codec_hiddens = torch.cat([last_id_hidden, pred_embeds.unsqueeze(0)], dim=1)
                    next_embed = codec_hiddens.sum(1, keepdim=True)
                    if generation_step < trailing_text.shape[1]:
                        next_embed = next_embed + trailing_text[:, generation_step : generation_step + 1, :]
                    else:
                        next_embed = next_embed + tts_pad_embed

                    step_t0 = time.perf_counter()
                    next_first_code = self.kernel.step_embed(next_embed)
                    if metrics is not None:
                        metrics.megakernel_step_ms += (time.perf_counter() - step_t0) * 1000.0
                    past_hidden = self.kernel.last_hidden_bf16
                    if generation_step == 0 and metrics is not None:
                        metrics.first_codec_ms = (time.perf_counter() - t0) * 1000.0
                    if generation_step % 4 == 0:
                        await asyncio.sleep(0)

                if metrics is not None:
                    elapsed = time.perf_counter() - t0
                    metrics.generation_ms = (time.perf_counter() - generation_t0) * 1000.0
                    metrics.codec_tokens_per_s = codebook_count / elapsed if elapsed > 0 else None
                if timing_logger is not None:
                    timing_logger.event("generation_done", codec_steps=codebook_count)

            except Exception:
                self.kernel.reset()
                raise

    async def stream_pcm_chunks(
        self,
        text: str,
        *,
        language: str = "English",
        ref_audio=None,
        ref_text: str | None = None,
        x_vector_only_mode: bool = True,
        code_chunk_steps: int = 6,
        left_context_steps: int = 10,
        metrics: TTSMetrics | None = None,
        log_path: str | os.PathLike | None = None,
        **kwargs,
    ) -> AsyncIterator[tuple[bytes, int]]:
        metrics = metrics or TTSMetrics()
        timing_logger = TimingLogger(log_path)
        pending: list[torch.Tensor] = []
        context: list[torch.Tensor] = []
        decode_start = time.perf_counter()
        total_t0 = time.perf_counter()
        first_audio_emitted = False
        timing_logger.event("stream_start", code_chunk_steps=code_chunk_steps, left_context_steps=left_context_steps)

        async for code in self.stream_codebooks(
            text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
            metrics=metrics,
            timing_logger=timing_logger,
            **kwargs,
        ):
            pending.append(code)  # keep on GPU; vocoder handles device move
            if len(pending) < code_chunk_steps:
                continue
            vocoder_t0 = time.perf_counter()
            pcm, sr = self._decode_window(context, pending)
            vocoder_ms = (time.perf_counter() - vocoder_t0) * 1000.0
            metrics.vocoder_ms += vocoder_ms
            if pcm:
                if not first_audio_emitted:
                    metrics.ttfc_ms = (time.perf_counter() - decode_start) * 1000.0
                    first_audio_emitted = True
                yield pcm, sr
                metrics.chunks += 1
                metrics.generated_audio_s += len(pcm) / 2 / sr
                timing_logger.event("audio_chunk", chunk=metrics.chunks, bytes=len(pcm), sample_rate=sr, vocoder_ms=vocoder_ms)
            context = (context + pending)[-left_context_steps:]
            pending = []

        if pending:
            vocoder_t0 = time.perf_counter()
            pcm, sr = self._decode_window(context, pending)
            vocoder_ms = (time.perf_counter() - vocoder_t0) * 1000.0
            metrics.vocoder_ms += vocoder_ms
            if pcm:
                if not first_audio_emitted:
                    metrics.ttfc_ms = (time.perf_counter() - decode_start) * 1000.0
                    first_audio_emitted = True
                yield pcm, sr
                metrics.chunks += 1
                metrics.generated_audio_s += len(pcm) / 2 / sr
                timing_logger.event("audio_chunk", chunk=metrics.chunks, bytes=len(pcm), sample_rate=sr, vocoder_ms=vocoder_ms, final=True)
        metrics.decode_s = time.perf_counter() - decode_start
        metrics.total_ms = (time.perf_counter() - total_t0) * 1000.0
        metrics.rtf = metrics.decode_s / metrics.generated_audio_s if metrics.generated_audio_s > 0 else None
        timing_logger.event("stream_done", **metrics.__dict__)

    def _decode_window(self, context: list[torch.Tensor], pending: list[torch.Tensor]) -> tuple[bytes, int]:
        codes = torch.stack(context + pending, dim=0).to(self.talker.device)
        wavs, sr = self.model.speech_tokenizer.decode([{"audio_codes": codes}])
        wav = wavs[0]
        if context:
            samples_per_code = int(self.model.speech_tokenizer.get_decode_upsample_rate())
            wav = wav[len(context) * samples_per_code :]
        return float_wav_to_pcm16(wav), sr
