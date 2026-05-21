# Qwen3-TTS Megakernel

Fast streaming TTS on RTX 5090 using a custom CUDA megakernel for Qwen3-TTS-0.6B. Plugs into [Pipecat](https://github.com/pipecat-ai/pipecat) as a drop-in `TTSService`.

## Performance (RTX 5090, warm)

| Metric | Value | Target |
|---|---:|---:|
| TTFC | **54 ms** | < 60 ms ✓ |
| RTF | **0.107** | < 0.15 ✓ |
| Prefill | **8 ms** | — |

> First call per speaker takes ~1.6 s to encode the reference audio. All subsequent calls use the cached speaker embedding and hit the numbers above.

## What it does

Qwen3-TTS has three inference stages: a 28-layer talker transformer, a 5-layer × 15-codebook predictor, and a waveform decoder. This repo replaces the first two with a single compiled CUDA megakernel, keeping the official package for audio I/O and the waveform decoder.

```
text + ref audio
  → voice prompt builder  (official qwen_tts)
  → megakernel talker     (28-layer, async prefill)
  → megakernel predictor  (5-layer × 15 codebooks, GPU-only AR loop)
  → waveform decoder      (official speech_tokenizer)
  → PCM chunks            (→ Pipecat TTSAudioRawFrame)
```

## Setup

Requirements: RTX 5090 / Blackwell GPU, CUDA 12.8+, Python 3.10+

```bash
git clone <this repo>
cd qwen_megakernel-master
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Clear any stale kernel cache if you get a build error:
```bash
rm -rf ~/.cache/torch_extensions/*/qwen_megakernel_C
```

## Usage

### CLI

```bash
python examples/stream_tts.py \
  --ref-audio /path/to/reference.wav \
  --text "Hello, streaming audio from Qwen3-TTS." \
  --out output.wav
```

### Streaming server

```bash
uvicorn qwen_megakernel.server:app --host 0.0.0.0 --port 8000
```

```bash
curl -N http://localhost:8000/stream_audio \
  -H "content-type: application/json" \
  -d '{"text": "Hello.", "ref_audio": "/path/to/reference.wav", "code_chunk_steps": 6}'
```

Audio arrives as NDJSON; each `"type": "audio"` line has base64-encoded PCM s16le at 24 kHz.

### Pipecat

```python
from qwen_megakernel.pipecat_service import QwenMegakernelTTSService
from qwen_megakernel.tts_stream import MegakernelQwen3TTS

engine = MegakernelQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
tts = QwenMegakernelTTSService(
    engine=engine,
    ref_audio="/path/to/reference.wav",
    code_chunk_steps=6,
)
```

Full pipeline demo (STT → LLM → TTS):
```bash
OPENAI_API_KEY=... python examples/pipecat_demo.py \
  --ref-audio /path/to/reference.wav \
  --text "What is the speed of light?"
```

## Tuning

| Parameter | Default | Effect |
|---|---:|---|
| `code_chunk_steps` | 6 | Codes per audio chunk. Lower = lower TTFC, higher RTF. |
| `left_context_steps` | 10 | Vocoder overlap window. Lower = faster, slightly less smooth. |
| `max_repeat` | 4 | Stop generation after N identical tokens in a row. |

## Notes

- The megakernel uses argmax decoding (no sampling). Speech style comes from the reference audio voice clone, not from temperature.
- `left_context_steps=10` is the recommended default. Increase to 20 for better chunk-boundary quality; decrease to 5 for minimum RTF.
- See `CHANGES.md` for the full optimization history and `BENCHMARK_RESULTS.md` for raw timing data.
