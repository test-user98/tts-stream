# Qwen3-TTS Megakernel — Optimizations & Techniques

How we took AlpinDale's Qwen3-0.6B text-decode megakernel and turned it into a
real-time streaming Qwen3-TTS backend — and the specific, non-obvious tricks that
got us from a 582 ms / RTF 1.22 first cut to **54 ms TTFC / RTF 0.107** on an RTX 5090.

> Companion docs: [BENCHMARK_METHODOLOGY.md](BENCHMARK_METHODOLOGY.md) (how the numbers
> are measured), [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) (the numbers),
> [CHANGES.md](CHANGES.md) (chronological journal incl. dead ends).

---

## 1. What we inherited (the reference we followed)

The starting point was AlpinDale's megakernel, which runs **Qwen3-0.6B text decode** as a
single persistent CUDA launch at ~1,000 tok/s on a 5090. We kept its core engine and the
techniques that make it fast:

- **One persistent kernel, 128 blocks × 512 threads** — the entire 28-layer forward pass
  runs in a single launch; the LM head + argmax is one fused follow-up kernel.
- **L1-bypassing 128-bit weight loads** (`ld.global.L1::no_allocate`) — weights stream
  through without evicting activations from cache.
- **Online (flash-style) softmax with vec4 KV reads** — no attention matrix is materialized.
- **RoPE via warp shuffles** — rotary embeddings stay in registers, zero shared-memory traffic.
- **Redundant RMSNorm on every block** — recomputing the norm everywhere is cheaper than the
  grid barrier it would otherwise need.
- **Idle blocks prefetch the next phase's weights into L2 during attention** — the kernel's
  signature trick; attention is low-bandwidth, so the other ~112 blocks warm upcoming weights.
- **Custom atomic grid barrier** (generation counter, no ABA race) replacing cooperative
  `grid.sync()`.

Reference material:
- Blog — https://blog.alpindale.net/posts/5090_decode_optimization/
- Megakernel — https://github.com/AlpinDale/qwen_megakernel
- Qwen3-TTS — https://github.com/QwenLM/Qwen3-TTS · https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
- Pipecat — https://docs.pipecat.ai

**Our job was integration, not kernel research** — but the integration surfaced several
genuinely new tricks below, and the kernel itself needed only a parameterization change
(no architecture change), because the TTS talker is the same Qwen3 block shape as 0.6B.

---

## 2. The baseline (first working integration)

After wiring the official `qwen_tts` package to the kernel:

| metric | value |
|---|---:|
| TTFC (warm) | 582 ms |
| RTF | 1.22 (slower than real time) |
| code predictor | **94 ms/step ← dominant bottleneck** |

The talker was already fast (~1.1 ms/step); everything downstream of it was the problem.

---

## 3. What we did — the smart bits

### 3.1 Run the 5-layer code predictor through the *same* compiled kernel
**Insight:** Qwen3-TTS's code predictor is a 5-layer Qwen3 backbone with the *identical block
shape* as the talker — so it doesn't need its own kernel. We drive it through the same
compiled `.so` by passing `num_layers=5` at runtime.

Two subtleties we had to solve:
- The kernel is compiled with a fixed `LDG_VOCAB_SIZE=3072` (the talker's codec head). The
  predictor's 15 per-codebook heads have a smaller vocab (~2048). **Fix:** zero-pad every
  lm_head *and* every embedding table to 3072 rows, and **clamp returned codes to the real
  vocab range**. Without the clamp, a zero-padded row could win the argmax when all real
  logits are negative and produce an out-of-bounds embedding lookup (CUDA assert).
- The actual attribute is `lm_head` (singular ModuleList), not `lm_heads` — we probe a list
  of candidate names so it's robust across `qwen_tts` versions.

**Result:** code predictor **94 ms → 1.16 ms/step (≈81×)**. TTFC 582 → 142 ms.

### 3.2 Fully-GPU autoregressive loop — zero CPU syncs
**Insight:** The predictor's 15-step inner loop originally called `.item()` after every token
to fetch the id for the next embedding lookup. Each `.item()` forces a full CUDA sync that
waits on all prior GPU work — 17 syncs × ~2.5 ms = ~42 ms of pure idle.

We made the loop never touch the CPU:
- `torch.index_select(embed_table, 0, out_token_long)` — GPU-side embedding gather.
- An int32→int64 `copy_()` mirror of the output token (`_out_token_long`), updated on the
  same CUDA stream, so the next step's gather is correctly ordered **without a sync**.

The single unavoidable sync is deferred to when the caller actually reads the codes.

**Result:** removed the idle sync time; predictor settles at ~1.16 ms for all 15 steps.

### 3.3 Batched codec-embedding lookup (one op instead of 15)
**Insight:** Computing the per-step codec hidden used 15 separate Python
`get_input_embeddings()[i](token)` calls — ~4 ms each, almost all Python→CUDA dispatch
overhead = ~57 ms/step.

We pre-stack all 15 embedding tables once into a single `[15, vocab, hidden]` tensor and do
one fancy-index op `stacked[arange(15), tokens]` per step.

**Result:** codec hidden computation **57 ms → ~1 ms/step**.

### 3.4 Async prefill (N syncs → 1)
**Insight:** Prefill ran the prompt through the talker with a `.item()` sync per token. Only
the *last* prefill token's output is needed (to read the first codec id for the EOS check).

We added `step_embed_async()` (fires the kernel, no sync) for the first N−1 tokens and one
synchronizing `step_embed()` on the last.

**Result:** prefill **~30 ms → 8 ms** (measured on real prompts).

### 3.5 Keep codec tensors on the GPU (no CPU round-trip)
**Insight:** the streaming loop was calling `.cpu()` on each yielded codec tensor, forcing a
sync + host round-trip, even though the vocoder accepts GPU tensors directly.

We keep codes on-device and let the vocoder's `.to(device)` be a no-op.

### 3.6 Shrink the vocoder window: `left_context_steps` 25 → 10
**Insight:** the streaming vocoder re-decodes `left_context + chunk_steps` codes every chunk
for waveform continuity. At 1.5 ms/code, 25+6 = 31 codes cost ~47 ms/chunk. Ten steps of
context (~0.83 s at 12 Hz) is enough for smooth boundaries.

**Result:** steady-state vocoder **~47 ms → ~24 ms**; RTF ~0.14 → ~0.09. *This is the single
biggest steady-state RTF lever* — the vocoder, not the kernel, dominates once decode is fast.

### 3.7 LM-head sized for the TTS vocab
**Insight:** the text model's LM head sweeps 151,936 rows; the TTS codec head is only 3,072.
We set `LDG_VOCAB_SIZE=3072` and drop the argmax grid to `LDG_LM_NUM_BLOCKS=128`
(ceil(3072/24)), so we don't waste SM occupancy scanning a vocab that isn't there.

### 3.8 Correct, deterministic termination (`max_repeat`)
**Observation:** in bf16 the megakernel's argmax can diverge from the reference in the
low-entropy tail and loop on one token instead of emitting EOS. We stop generation after
`max_repeat` identical first-codebook tokens — mirroring how the reference's EOS terminates
the silent tail. (Correctness on the *voiced* region is checked separately via the WER
harness — see §5.)

### 3.9 Productionization that costs nothing on the happy path
`warm_speaker()` pre-encodes the reference voice at startup (kills the ~1.6 s cold start);
an `asyncio.Lock` serializes the stateful KV cache; `kernel.reset()` on any exception keeps a
failed request from corrupting the next; a wall-clock generation timeout and ref-audio
validation fail fast. All are zero-cost when uncontended/healthy.

---

## 4. Result

| metric | baseline | final | change |
|---|---:|---:|---|
| **TTFC (warm)** | 582 ms | **54 ms** | **~11× / under 60 ms target** |
| **RTF** | 1.22 | **0.107** | **faster than real time** |
| prefill | ~30 ms | **8 ms** | ~3.7× |
| code predictor | 94 ms/step | **1.16 ms/step** | **~81×** |
| talker decode | ~1.1 ms/step | ~1.1 ms/step | at kernel spec |

vs. the unmodified `qwen_tts` path (RTF ≈ 3.55) the pipeline is **~33× faster end-to-end**.

---

## 5. Ideas for further gains (bonus / next)

- **Streaming vocoder with cached conv state** — today we re-decode the `left_context`
  overlap every chunk. Caching the vocoder's convolution state would remove that redundant
  compute entirely — the largest remaining honest RTF win.
- **CUDA-graph the per-step launch chain** (talker step → LM head → predictor's 15 steps) to
  erase per-launch overhead — a clean "we improved the megakernel during integration" win.
- **Cheaper/fewer grid barriers** — the blog names the 4 atomic barriers/layer (~8.8 µs) as
  the wall; the deepest kernel-level win, and the highest risk.

---

## 6. Dead ends (so they're not re-tried)

- `torch.compile` on the speech tokenizer — 20+ min inductor compile on first run, blocked
  inference. Left as a manual opt-in.
- Using `_out_token` (int32) directly as a long index — wrong dtype; needed the int64 mirror.
- Per-chunk `setframerate()` in the CLI — `wave` errors if params change mid-write; set once.
