# Benchmark Methodology

How the numbers in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) were measured, and how to
reproduce them. Every figure comes directly from the `TTSMetrics` instrumentation in
[qwen_megakernel/tts_stream.py](qwen_megakernel/tts_stream.py); the benchmark harness only
drives the engine and aggregates timings — it introduces no measurement logic of its own.

> For *what* we optimized and the techniques behind these numbers (the smart tricks, plus the
> reference work we followed), see **[OPTIMIZATIONS.md](OPTIMIZATIONS.md)**.

## 1. Environment

| | |
|---|---|
| GPU | NVIDIA RTX 5090 (Blackwell, sm_120a), rented on Vast.ai |
| CUDA | 12.8+ |
| PyTorch | 2.7 |
| Model | `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, bf16, no quantization |
| Kernel | JIT-compiled with `-arch=sm_120a` (see [build.py](qwen_megakernel/build.py)) |
| Decoding | argmax (greedy), single stream, batch size 1 |

The megakernel replaces only the **28-layer talker** forward pass and drives the
**5-layer code predictor** through the same compiled kernel. The official `qwen_tts`
package still owns text/audio preprocessing, speaker encoding, and the waveform vocoder.

## 2. Metric definitions

All timings use `time.perf_counter()` (monotonic wall clock).

- **TTFC (time to first audio chunk)** — wall-clock from the start of
  `stream_pcm_chunks()` until the **first non-empty PCM chunk is yielded**.
  This window includes: prompt preparation, talker prefill, the first `chunk_steps`
  talker+code-predictor decode steps, and the first vocoder decode. It is set once,
  at `metrics.ttfc_ms`, when the first chunk is emitted
  ([tts_stream.py:451](qwen_megakernel/tts_stream.py#L451)). This is the metric the
  assignment targets.

- **RTF (real-time factor)** — `decode_s / generated_audio_s`
  ([tts_stream.py:475](qwen_megakernel/tts_stream.py#L475)), where `decode_s` is the
  total wall-time of the whole streamed synthesis and `generated_audio_s` is the
  duration of audio produced (`bytes / 2 / sample_rate`, since PCM is s16le mono).
  RTF < 1 means synthesis is faster than playback.

- **Prefill** — wall-time of the talker prefill loop over the prompt embeddings
  ([tts_stream.py:322‑333](qwen_megakernel/tts_stream.py#L322-L333)). Uses async
  dispatch for the first N−1 prompt tokens and one synchronizing step on the last
  token (to read `first_code` for the EOS check).

- **Cold vs. warm** — the first request for a given speaker pays a one-time
  reference-audio encode (voice-prompt **cache miss**, ~1.6 s). Every later request
  with the same speaker hits the cache (`_voice_prompt_cache`). "Warm" numbers are
  steady state; `warm_speaker()` pre-populates the cache at startup so production
  never serves a cold request.

- **Codec step** — one talker decode step → one set of 16 codec codes
  (1 first-codebook token from the talker + 15 from the code predictor). At 12 Hz,
  one codec step ≈ 1/12 s of audio.

## 3. Run protocol

Implemented in [qwen_megakernel/bench_tts.py](qwen_megakernel/bench_tts.py):

1. Load the model once.
2. **Cold run**: `clear_voice_prompt_cache()`, then one synthesis. The reported cold
   TTFC therefore includes the speaker-encode cost.
3. **Warm runs**: the speaker is now cached. Discard `--warmup` runs (default 3) to
   settle JIT/cuDNN/allocator state, then measure `--runs` runs (default 10).
4. Report per-run JSON plus aggregates: **TTFC p50 / p95 / min**, **RTF mean / p95**,
   mean prefill. Percentiles (not a single best run) are the honest summary.

Reproduce the headline configuration:

```bash
python -m qwen_megakernel.bench_tts \
    --ref-audio /path/to/clone.wav \
    --ref-text "transcript of the reference clip" \
    --text "Hello." \
    --warmup 3 --runs 10 \
    --chunk-steps 6 --left-context-steps 10
```

Sweep the latency/throughput trade-off with `--sweep-chunk-steps 4,6,8,12`.

## 4. Baseline comparison

The reference figure (RTF ≈ 3.55, i.e. ~33× slower than this pipeline) is the
unmodified `qwen_tts` generation path on the same GPU and model, measured by
`model.generate_voice_clone(...)` end-to-end. The megakernel's per-component speedups
(talker ~1.1 ms/step, code predictor 1.16 ms/step vs ~94 ms/step for HF
`code_predictor.generate()`) are recorded in [CHANGES.md](CHANGES.md) and exposed as
`code_predictor_ms` / `megakernel_step_ms` in `TTSMetrics`.

## 5. Honest caveats

These do not invalidate the numbers but are stated so the methodology is reproducible
and the figures are interpreted correctly:

- **Utterance length.** The headline TTFC/RTF were measured on a short utterance.
  TTFC is length-independent (it's time to the *first* chunk), but RTF should also be
  reported for longer, real sentences. Use `--text "<a real sentence>"`.
- **RTF denominator includes the silence-padded tail.** Generation stops on a
  repetition heuristic (`max_repeat`, see [CHANGES.md](CHANGES.md) Step 8), so the
  tail can contain near-silent frames. Both the wall-time to produce them (numerator)
  and their duration (denominator) are counted, so RTF is a true end-to-end ratio, but
  it is not a "voiced-audio-only" RTF.
- **argmax only.** The megakernel does greedy decode; speaker style comes from the
  voice-clone embedding, not sampling.

## 6. Correctness validation (separate from speed)

Speed numbers are only meaningful if the audio is correct. Correctness is verified
independently with [examples/verify_correctness.py](examples/verify_correctness.py),
which synthesizes each sentence both via the reference `generate_voice_clone` path and
via the megakernel, transcribes both with Whisper, and compares word error rate
(WER) against the input text. A small, length-stable gap between megakernel-WER and
reference-WER indicates the kernel reproduces the model; a growing gap would indicate a
kernel divergence. Run it and record the summary alongside the speed table.
```bash
python examples/verify_correctness.py --ref-audio clone.wav --ref-text "..." --out-dir verify_out
```
