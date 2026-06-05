"""Reproducible TTS benchmark for the Qwen3-TTS megakernel.

Produces the TTFC / RTF / prefill numbers reported in BENCHMARK_RESULTS.md.
Every metric is taken straight from the TTSMetrics instrumentation in
qwen_megakernel/tts_stream.py -- this script only drives the engine, times
runs, and aggregates; it adds no measurement logic of its own.

Definitions (see BENCHMARK_METHODOLOGY.md for the full writeup):
  TTFC  time-to-first-chunk: wall-clock from the start of stream_pcm_chunks()
        until the first non-empty PCM chunk is yielded (includes prompt prep,
        prefill, the first `chunk_steps` decode steps, and the first vocoder call).
  RTF   real-time factor: total generation wall-time / seconds of audio produced.
        RTF < 1 means faster than real time.
  cold  first request for a speaker -- includes the one-time reference-audio
        encode (voice-prompt cache miss).
  warm  steady state -- speaker already cached (warm_speaker called first).

USAGE (RTX 5090, qwen_tts + model installed):

    python -m qwen_megakernel.bench_tts \
        --ref-audio /path/to/clone.wav \
        --ref-text "transcript of the reference clip" \
        --runs 10 --warmup 3 --sweep-chunk-steps 4,6,8,12

Writes raw per-run JSON lines and an aggregate table; pass --out results.jsonl
to also persist them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from qwen_megakernel.tts_stream import MegakernelQwen3TTS, TTSMetrics

DEFAULT_TEXT = "Hello."


async def _one_run(engine, text, *, language, ref_audio, ref_text,
                   x_vector_only_mode, chunk_steps, left_context_steps):
    """Drain one full synthesis and return its TTSMetrics."""
    metrics = TTSMetrics()
    async for _pcm, _sr in engine.stream_pcm_chunks(
        text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only_mode,
        code_chunk_steps=chunk_steps,
        left_context_steps=left_context_steps,
        metrics=metrics,
    ):
        pass  # discard audio; we only want the timings
    return metrics


def _row(run_label, text, chunk_steps, left_context_steps, m: TTSMetrics) -> dict:
    return {
        "run": run_label,
        "ttfc_ms": round(m.ttfc_ms, 1) if m.ttfc_ms is not None else None,
        "rtf": round(m.rtf, 4) if m.rtf is not None else None,
        "audio_s": round(m.generated_audio_s, 3),
        "prefill_ms": round(m.prefill_ms, 1) if m.prefill_ms is not None else None,
        "codec_steps": m.codec_steps,
        "chunks": m.chunks,
        "prompt_prepare_ms": round(m.prompt_prepare_ms, 1) if m.prompt_prepare_ms else None,
        "vocoder_ms": round(m.vocoder_ms, 1),
        "code_predictor_ms": round(m.code_predictor_ms, 2),
        "megakernel_step_ms": round(m.megakernel_step_ms, 2),
        "text": text,
        "chunk_steps": chunk_steps,
        "left_context": left_context_steps,
    }


def _agg(label, rows: list[dict]) -> dict:
    ttfc = np.array([r["ttfc_ms"] for r in rows if r["ttfc_ms"] is not None], dtype=float)
    rtf = np.array([r["rtf"] for r in rows if r["rtf"] is not None], dtype=float)
    pre = np.array([r["prefill_ms"] for r in rows if r["prefill_ms"] is not None], dtype=float)
    return {
        "config": label,
        "n": len(rows),
        "ttfc_p50_ms": round(float(np.percentile(ttfc, 50)), 1) if ttfc.size else None,
        "ttfc_p95_ms": round(float(np.percentile(ttfc, 95)), 1) if ttfc.size else None,
        "ttfc_min_ms": round(float(ttfc.min()), 1) if ttfc.size else None,
        "rtf_mean": round(float(rtf.mean()), 4) if rtf.size else None,
        "rtf_p95": round(float(np.percentile(rtf, 95)), 4) if rtf.size else None,
        "prefill_mean_ms": round(float(pre.mean()), 1) if pre.size else None,
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--ref-text")
    ap.add_argument("--language", default="English")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="Utterance to synthesize (use a real sentence for honest RTF).")
    ap.add_argument("--warmup", type=int, default=3, help="Warm runs discarded before measuring.")
    ap.add_argument("--runs", type=int, default=10, help="Measured warm runs.")
    ap.add_argument("--chunk-steps", type=int, default=6)
    ap.add_argument("--left-context-steps", type=int, default=10)
    ap.add_argument("--sweep-chunk-steps", help="Comma-separated, e.g. 4,6,8,12. Overrides --chunk-steps.")
    ap.add_argument("--x-vector-only", action="store_true")
    ap.add_argument("--out", help="Append raw per-run JSON lines here.")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    engine = MegakernelQwen3TTS.from_pretrained(args.model)

    chunk_list = (
        [int(v) for v in args.sweep_chunk_steps.split(",") if v.strip()]
        if args.sweep_chunk_steps else [args.chunk_steps]
    )

    out_fh = open(args.out, "a", encoding="utf-8") if args.out else None
    raw_rows: list[dict] = []
    aggregates: list[dict] = []

    def emit(row: dict) -> None:
        raw_rows.append(row)
        line = json.dumps(row, sort_keys=True)
        print(line)
        if out_fh:
            out_fh.write(line + "\n")

    kw = dict(language=args.language, ref_audio=args.ref_audio, ref_text=args.ref_text,
              x_vector_only_mode=args.x_vector_only)

    for chunk_steps in chunk_list:
        # ---- COLD: clear the voice-prompt cache so the speaker is re-encoded ----
        engine.clear_voice_prompt_cache()
        t0 = time.perf_counter()
        m = await _one_run(engine, args.text, chunk_steps=chunk_steps,
                           left_context_steps=args.left_context_steps, **kw)
        cold = _row(f"cold_cs{chunk_steps}", args.text, chunk_steps, args.left_context_steps, m)
        cold["wall_total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        emit(cold)

        # ---- WARM: speaker now cached; warm up, then measure ----
        for _ in range(args.warmup):
            await _one_run(engine, args.text, chunk_steps=chunk_steps,
                           left_context_steps=args.left_context_steps, **kw)
        warm_rows = []
        for i in range(args.runs):
            m = await _one_run(engine, args.text, chunk_steps=chunk_steps,
                               left_context_steps=args.left_context_steps, **kw)
            r = _row(f"warm_cs{chunk_steps}_{i}", args.text, chunk_steps, args.left_context_steps, m)
            emit(r)
            warm_rows.append(r)
        aggregates.append(_agg(f"warm chunk_steps={chunk_steps}", warm_rows))

    if out_fh:
        out_fh.close()

    print("\n" + "=" * 78)
    print(f"AGGREGATE  (text={args.text!r}, warmup={args.warmup}, runs={args.runs})")
    print("-" * 78)
    print(f"{'config':<26}{'TTFC p50':>10}{'TTFC p95':>10}{'TTFC min':>10}{'RTF mean':>10}{'prefill':>10}")
    for a in aggregates:
        print(f"{a['config']:<26}{a['ttfc_p50_ms']:>10}{a['ttfc_p95_ms']:>10}"
              f"{a['ttfc_min_ms']:>10}{a['rtf_mean']:>10}{a['prefill_mean_ms']:>10}")
    print("=" * 78)
    print("TTFC = wall-clock to first audio chunk; RTF = gen wall-time / audio seconds (<1 = realtime).")
    print("Cold rows include the one-time speaker encode; warm rows are steady state.")


if __name__ == "__main__":
    asyncio.run(main())
