"""Smoke test for streaming Qwen3-TTS audio chunks.

Example:
    python examples/stream_tts.py --ref-audio clone.wav --text "Hello from the megakernel."
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import wave

from qwen_megakernel.tts_stream import MegakernelQwen3TTS, TTSMetrics


async def main():
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text")
    parser.add_argument("--out", default="qwen3_tts_megakernel_stream.wav")
    parser.add_argument("--log-jsonl", help="Write detailed timing events as JSONL.")
    parser.add_argument("--chunk-steps", type=int, default=6)
    parser.add_argument(
        "--sweep-chunk-steps",
        help="Comma-separated chunk sizes to benchmark, e.g. 4,6,8,12,16. Writes one wav per size.",
    )
    parser.add_argument("--icl", action="store_true", help="Use ref_text/ref_code ICL mode instead of speaker-vector-only mode.")
    args = parser.parse_args()

    engine = MegakernelQwen3TTS.from_pretrained(args.model)

    chunk_steps_list = (
        [int(v.strip()) for v in args.sweep_chunk_steps.split(",") if v.strip()]
        if args.sweep_chunk_steps
        else [args.chunk_steps]
    )

    for chunk_steps in chunk_steps_list:
        metrics = TTSMetrics()
        out_path = args.out
        if len(chunk_steps_list) > 1:
            stem, dot, suffix = args.out.rpartition(".")
            out_path = f"{stem or args.out}.chunk{chunk_steps}.{suffix or 'wav'}"
        log_path = args.log_jsonl
        if log_path and len(chunk_steps_list) > 1:
            stem, dot, suffix = log_path.rpartition(".")
            log_path = f"{stem or log_path}.chunk{chunk_steps}.{suffix or 'jsonl'}"

        sample_rate = int(engine.model.speech_tokenizer.get_output_sample_rate())
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            async for pcm, sr in engine.stream_pcm_chunks(
                args.text,
                language=args.language,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                x_vector_only_mode=not args.icl,
                code_chunk_steps=chunk_steps,
                metrics=metrics,
                log_path=log_path,
            ):
                wf.writeframes(pcm)
                print(f"chunk_steps={chunk_steps} chunk={metrics.chunks + 1}: {len(pcm)} bytes @ {sr} Hz")

        print(json.dumps({"chunk_steps": chunk_steps, "out": out_path, "log_jsonl": log_path, **dataclasses.asdict(metrics)}, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
