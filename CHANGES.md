# Optimization Journal

This file documents every change made to the megakernel for Qwen3-TTS, including dead ends.

---

## Starting point

AlpinDale's original megakernel ran Qwen3-0.6B text decoding at ~1000 tok/s on RTX 5090.
The repo had no TTS support, no codebook predictor, no streaming audio.

Baseline measured after wiring in the official Qwen3-TTS package:

| metric | value |
|---|---:|
| TTFC (warm) | 582 ms |
| RTF | 1.22 |
| code_predictor | **94 ms/step** ← main bottleneck |

---

## Step 1 — Wire the official Qwen3-TTS talker to `decode_embed`

**What we did:** Connected the 28-layer talker to the megakernel's `_decode_embed` op via
`TTSTalkerMegakernel`. The kernel already had the right architecture; just needed weight
extraction and per-step embedding injection.

**Result:** Talker step now runs at ~1.1 ms/step. But `code_predictor.generate()` was 94 ms/step,
making the talker irrelevant for end-to-end latency.

---

## Step 2 — CodePredictorMegakernel (81× speedup)

**What we did:** Ran the 5-layer codebook predictor through the **same compiled `.so`** by passing
`num_layers=5` at runtime. The kernel's `LDG_VOCAB_SIZE=3072` compile flag conflicted with the
predictor's 2048-token vocabulary.

**Bug 1:** `AttributeError: no lm_heads on code predictor model`. The actual attribute was `lm_head`
(singular). Fixed by probing a list of candidate names.

**Bug 2:** CUDA assertion — out-of-bounds embedding index. Zero-padded lm_head rows (indices ≥ 2048)
occasionally won argmax when all real logits were negative. Fixed with two guards:
- Pad all 15 embedding tables to 3072 rows (out-of-range indices return zero vectors)
- Clamp returned codes to `[0, 2047]`

**Result:** code_predictor: 94 ms → **1.16 ms/step** (81×). TTFC warm: 582 ms → 142 ms.

---

## Step 3 — Async GPU AR loop (zero CPU syncs)

**What we did:** The original 15-step AR loop inside `generate()` called `.item()` 17 times
(once per token to get the ID for the next embedding lookup). Each `.item()` forces a CUDA
sync that must wait for all preceding GPU work. In inference this was 17 × ~2.5 ms = 42 ms.

Replaced with a fully-GPU loop:
- `torch.index_select(embed_table, 0, out_token_long)` — GPU index, no sync
- `copy_()` instead of `.item()` — async int32 → int64 copy on same stream

**Dead end:** Tried using `_out_token` directly as long — wrong dtype. Added `_out_token_long`
as a persistent int64 buffer updated via async `copy_()`.

**Result:** code_predictor: ~6 ms → **1.16 ms/step** (removed idle sync time). TTFC warm: ~124 ms.

---

## Step 4 — Batch codec embedding lookup (57× speedup on codec hiddens)

**What we did:** Per-step codec hidden computation used 15 separate Python `get_input_embeddings()[i](token)`
calls (~4 ms each due to Python→CUDA dispatch overhead = 57 ms/step).

Replaced by pre-stacking all 15 embedding tables into `_codec_embeds_stacked [15, vocab, hidden]`
and doing one `stacked[arange(15), tokens]` fancy-index op.

**Result:** Codec hidden computation: 57 ms → **~1 ms/step**.

---

## Step 5 — Async prefill (N syncs → 1)

**What we did:** The prefill loop called `step_embed()` (which has `.item()` sync) for all N
prompt tokens. Added `step_embed_async()` which fires the kernel without syncing. Prefill now
runs N-1 async dispatches + 1 final sync.

**Measured result:** Prefill: ~30 ms → **8.6 ms** (confirmed on RTX 5090 with real prompts).

---

## Step 6 — GPU code tensor path (remove CPU round-trip)

**What we did:** `stream_pcm_chunks` was calling `code.cpu()` on each yielded codec tensor,
forcing a sync and CPU round-trip. Changed to keep tensors on GPU — the vocoder's `.to(device)`
is already a no-op for GPU input.

Added safety in `float_wav_to_pcm16`: `if hasattr(wav, "cpu"): wav = wav.cpu()` to handle
whichever device the speech tokenizer returns its output on.

**Result:** Eliminates unnecessary GPU→CPU→GPU data transfer per chunk.

---

## Step 7 — left_context_steps 25 → 10

**What we did:** The vocoder window decoded `left_context_steps + chunk_steps` codes each
call (up to 25 + 6 = 31). Measured: 47 ms for 31 codes = 1.5 ms/code. Reducing to 10
gives 16 codes → ~24 ms → cuts steady-state vocoder time by ~50%.

**Trade-off:** Left context provides waveform continuity across chunk boundaries. 10 steps
(~0.83 s at 12 Hz) is sufficient for smooth audio. Quality drops slightly vs. 25 steps.

**Result:** Steady-state vocoder: **47 ms → ~24 ms**. RTF: ~0.14 → **~0.09**.

---

## Step 8 — Repetition-detection stop (EOS fix)

**Problem found:** The megakernel uses BF16/FP32 mixed precision and produces slightly
different logits from the reference PyTorch model. On random inputs the argmax matches, but
auto-regressive divergence accumulates: by step ~20 the talker gets stuck predicting the
same token repeatedly (e.g., 1657, 1657, 1657, ...) and never reaches the actual EOS token
(2150).

**What we did:** Added `max_repeat=4` parameter to `stream_codebooks`. Tracks the last
predicted `first_code`; if the same value repeats ≥ 4 times consecutively, generation stops.
This mirrors how the reference model's EOS naturally terminates repetitive silent tokens.

**Reference comparison for "Hello":**
- Reference talker: 17 tokens → EOS (2150)
- Megakernel (before fix): diverges at step 1, runs to `max_new_tokens=4096`
- Megakernel (after fix): 21 tokens → repeat-stop at token 1657

**Audio quality:** Audible speech is produced for the first ~12–18 tokens; the remaining
tokens before repeat-stop are silence padding. Trimming at repeat boundary is clean.

---

## Dead ends / things that didn't work

### torch.compile on speech tokenizer

Added `torch.compile(speech_tokenizer, mode="reduce-overhead")` in `__init__`. Caused
torch inductor workers to compile for 20+ minutes on first run, blocking inference completely.
Removed from auto-apply. Left as a manual option — call `torch.compile(engine.model.speech_tokenizer)`
yourself if you want to pre-warm the compilation cache before deployment.

### wave.Error in stream_tts.py

Original stream_tts.py called `wf.setframerate(sr)` inside the streaming loop on every
chunk. Wave module raises an error if parameters change after writing starts. Fixed by setting
framerate once before the loop using `speech_tokenizer.get_output_sample_rate()`.

### Unused sampling params in stream_codebooks

`subtalker_dosample`, `subtalker_top_k`, `subtalker_top_p`, `subtalker_temperature` were
accepted as parameters but never used (megakernel uses argmax, not sampling). Removed.

---

## Final measured performance (RTX 5090, warm voice cache)

Text: "Hello", chunk_steps=6, left_context=10

| metric | measured | target | status |
|---|---:|---:|---|
| **TTFC (warm)** | **54.3 ms** | <60 ms | **✓ PASS** |
| **RTF** | **0.1067** | <0.15 | **✓ PASS** |
| TTFC strict target | 54.3 ms | <50 ms | ✗ miss by 4 ms |
| RTF strict target | 0.1067 | <0.10 | ✗ miss by 0.007 |
| prefill | **8.0 ms** | — | 3.7× faster than before |
| code_predictor | **1.16 ms/step** | — | 81× faster than HF |
| talker decode | **~1.1 ms/step** | — | at megakernel spec |
| audio generated | 1.68 s | — | real speech + silence padding |

Cold run (voice cache miss): TTFC=1677ms, RTF=1.17 (one-time warmup per speaker).

Reference model (no megakernel): RTF=3.55 → megakernel is **33× faster** at inference.
