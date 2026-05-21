# Benchmark Results — Qwen3-TTS Megakernel

Hardware: NVIDIA RTX 5090 (Vast.ai), CUDA 12+, PyTorch 2.7
Model: Qwen/Qwen3-TTS-12Hz-0.6B-Base
Date: 2026-05-21

## Performance Targets
| Metric | General Target | Strict Target |
|--------|---------------|--------------|
| TTFC   | < 60 ms       | < 50 ms      |
| RTF    | < 0.15        | < 0.10       |

## Measured Results

### Short text ("Hello"), chunk_steps=6, left_context=10

| Run | TTFC (ms) | RTF | Audio (s) | Prefill (ms) | Codec steps |
|-----|----------|-----|-----------|-------------|-------------|
| Cold (voice cache miss) | 1677.1 | 1.1728 | 1.68 | 8.6 | 21 |
| **Warm (voice cached)** | **54.3** | **0.1067** | 1.68 | 8.0 | 21 |

### Notes
- Cold TTFC includes ~1.5s voice prompt cache build (one-time per speaker)
- Warm TTFC is the steady-state latency for repeated inference with same speaker
- Prefill time: 8.0–8.6ms (async prefill optimization, down from ~30ms baseline)
- Generation stops via repetition detection (megakernel diverges from reference EOS due to numerical precision differences)
- Reference model generates 1.04s of audio in 3.70s (RTF 3.554) without the megakernel
- Megakernel provides ~33× inference speedup over reference (RTF 0.107 vs 3.554)

## Target Status
| Metric | Measured | General Target (< 60ms / < 0.15) | Strict Target (< 50ms / < 0.10) |
|--------|---------|----------------------------------|----------------------------------|
| TTFC   | 54.3 ms | **PASS** | MISS (by 4ms) |
| RTF    | 0.1067  | **PASS** | MISS (by 0.007) |

## Optimization History
| Optimization | TTFC Impact | RTF Impact |
|---|---|---|
| Baseline (before this session) | ~90ms est. | ~0.25 est. |
| Async prefill (N-1 async + 1 sync) | prefill 30ms → 8.6ms | minor |
| GPU code tensor (no CPU round-trip) | — | ~5% faster |
| left_context_steps 25 → 10 | — | ~50% faster vocoder |
| Repetition-detection stop (EOS fix) | enables correct termination | — |

## Raw Timing Data (bench_final.py, warm run 2)
```json
{"run": "warm_cs6", "ttfc_ms": 54.3, "rtf": 0.1067, "audio_s": 1.68,
 "prefill_ms": 8.0, "codec_steps": 21, "chunks": 4,
 "text": "Hello", "chunk_steps": 6, "left_context": 10}
```
```json
{"run": "cold_cs6", "ttfc_ms": 1677.1, "rtf": 1.1728, "audio_s": 1.68,
 "prefill_ms": 8.6, "codec_steps": 21, "chunks": 4,
 "text": "Hello", "chunk_steps": 6, "left_context": 10}
```
