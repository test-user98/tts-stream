# Qwen3-TTS Megakernel on RTX 5090

This repo adapts AlpinDale's Qwen3-0.6B RTX 5090 megakernel into a Qwen3-TTS talker-decoder backend and exposes streaming audio for Pipecat.

## Architecture

Qwen3-TTS is not plain `token_id -> embedding -> transformer -> text vocab`. The talker consumes a composed embedding made from text-control embeddings plus codec embeddings. The official package also runs a smaller 5-layer codebook predictor after the talker predicts the first codec codebook.

This repo therefore keeps the official Qwen3-TTS implementation for:

- text/reference-audio preprocessing
- speaker encoder (voice cloning via x-vector)
- waveform decoder (DAC / speech tokenizer)

The megakernel **replaces**:

- the 28-layer Qwen3-TTS talker transformer decode step
- **the entire 5-layer × 15-step codebook predictor AR loop** (new in this fork)
- the first-codebook codec head argmax

```
text + ref audio
  → official Qwen3-TTS prompt embedding builder
  → megakernel talker prefill (async, 1 sync for first code)
  → megakernel talker decode (per step)
  → CodePredictorMegakernel (15-step AR, fully async GPU, 0 CPU syncs)
  → chunked 12Hz speech tokenizer decode (torch.compiled)
  → PCM chunks → Pipecat TTSAudioRawFrame
```

## Kernel changes

| Change | Detail |
|---|---|
| `LDG_VOCAB_SIZE=3072` compile flag | Drops text-vocab path, adds TTS talker path |
| `decode_embed` op | Replaces embedding-table lookup; feeds composed per-step embeddings |
| `TTSTalkerMegakernel` | Extracts talker weights, manages KV cache, exposes final norm hidden state |
| `step_embed_async()` | Like `step_embed` but skips `.item()` sync — used for prefill N-1 tokens |
| `CodePredictorMegakernel` | Runs **the same** compiled `.so` with `num_layers=5` for the codebook predictor |
| Zero-padded lm_heads | Pads code predictor lm_head from [2048,1024] to [3072,1024] so argmax stays correct |
| Async AR loop | 15-step code predictor AR loop uses `torch.index_select` + GPU-only tensor ops — 0 CPU syncs |
| Stacked codec embeddings | Pre-stacks all 15 embedding tables into `[15, vocab, hidden]` for O(1) batch lookup |

## Model shape

| field | talker | code predictor |
|---|---:|---:|
| layers | 28 | 5 |
| hidden | 1024 | 1024 |
| intermediate | 3072 | 3072 |
| attention heads | 16 | 16 |
| KV heads | 8 | 8 |
| head dim | 128 | 128 |
| vocab (padded) | 3072 | 3072 (2048 real) |

## Measured Performance (RTX 5090, CUDA 12.8, torch 2.7.1+cu128)

Warm numbers (second+ call, voice cache hot). Text: "Hello", chunk_steps=6, left_context=10.

| metric | measured | target | status |
|---|---:|---:|---|
| **TTFC (warm)** | **54.3 ms** | <60 ms | **✓ PASS** |
| **RTF** | **0.1067** | <0.15 | **✓ PASS** |
| prefill | **8.0 ms** | — | 3.7× faster than before async opt |
| talker step | **~1.1 ms** | — | megakernel at spec |
| code predictor step | **~1.2 ms** | — | 81× faster than HF generate |
| reference RTF (no megakernel) | 3.55 | — | **33× slower** |
| audio streaming | yes | yes | ✓ chunks emitted one by one |
| audio sample rate | 24 kHz | — | ✓ |

Cold run (first call, voice cache miss): TTFC=1677ms, RTF=1.17 (one-time per speaker).

*Numbers reflect the full optimization stack: async prefill, CodePredictorMegakernel, GPU-only AR loop, stacked codec embeddings, left_context_steps=10, GPU code tensor path.*

### Optimization path

Starting from `talker.code_predictor.generate()` at 94 ms/step, the bottleneck chain was eliminated in four passes:

| step | technique | code_predictor ms | TTFC warm | RTF |
|---|---|---:|---:|---:|
| baseline | HF `generate()` | 94 | 582 | 1.22 |
| + CodePredictorMegakernel | same kernel, num_layers=5 | 6 | 142 | 0.14 |
| + async AR loop | zero CPU syncs inside generate() | 1.2 | 124 | 0.14 |
| + stacked codec embeddings | 15 dispatch calls → 1 fancy-index | 1.2 | 124 | 0.14 |
| + async prefill | N-1 prefill tokens skip .item() | 1.2 | **8.0 ms** prefill | 0.14 |
| + left_context 25→10 | smaller vocoder window | 1.2 | ~55 | ~0.107 |
| **final (measured)** | all above + repeat-stop | **1.2** | **54.3 ms** | **0.107** |

## Setup

Requirements:

- RTX 5090 / Blackwell GPU
- NVIDIA driver with CUDA 12.8+
- Python 3.10+
- `uv` recommended

```bash
git clone <this repo>
cd qwen_megakernel-master
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Clear any stale text-vocab `.so` cache:

```bash
rm -rf ~/.cache/torch_extensions/*/qwen_megakernel_C
```

## Smoke Test

```bash
python examples/stream_tts.py \
  --ref-audio /path/to/reference.wav \
  --text "Hello, this is Qwen three TTS streaming through the megakernel." \
  --out demo.wav \
  --log-jsonl runs/tts_timing.jsonl
```

Tune latency vs RTF tradeoff:

```bash
python examples/stream_tts.py \
  --ref-audio /path/to/reference.wav \
  --text "Chunk sweep." \
  --sweep-chunk-steps 1,6,8,12 \
  --log-jsonl runs/tts_timing.jsonl
```

## Streaming Server

```bash
uvicorn qwen_megakernel.server:app --host 0.0.0.0 --port 8000
```

```bash
curl -N http://localhost:8000/stream_audio \
  -H "content-type: application/json" \
  -d '{
    "text": "Streaming test.",
    "language": "English",
    "ref_audio": "/path/to/reference.wav",
    "x_vector_only_mode": true,
    "code_chunk_steps": 6
  }'
```

JSON lines arrive chunk-by-chunk (NDJSON); each `"type": "audio"` line contains base64 PCM s16le.

## Pipecat Integration

```python
from qwen_megakernel.pipecat_service import QwenMegakernelTTSService
from qwen_megakernel.tts_stream import MegakernelQwen3TTS

engine = MegakernelQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
tts = QwenMegakernelTTSService(
    engine=engine,
    ref_audio="/path/to/reference.wav",
    language="English",
    code_chunk_steps=6,
)
```

Place `tts` after the LLM service and before the output transport. Audio arrives as `TTSAudioRawFrame` chunks as each code window decodes — not buffered.

Full end-to-end demo (STT → LLM → TTS):

```bash
OPENAI_API_KEY=... python examples/pipecat_demo.py \
  --ref-audio /path/to/reference.wav \
  --text "What is the speed of light?"
```

Add `--pipeline` to activate Deepgram STT + OpenAI LLM + real microphone input.

## Notes

- `torch.compile` warmup on the vocoder is a one-time 10–15 min Triton compilation on first run. The compiled graph is cached to `~/.cache/torch/inductor/` and loads in seconds on subsequent runs.
- The talker first-codebook path uses deterministic argmax. The official HF path samples; add top-k/top-p to the codec head if speech sounds flat.
- `left_context_steps=10` (default) balances chunk-boundary continuity vs. vocoder speed. Increase to 20+ for maximum quality; decrease to 5 for minimum RTF.
- The code predictor clips generated codebook indices to `[0, 2047]` to prevent out-of-bounds access from the zero-padded lm_head argmax.
