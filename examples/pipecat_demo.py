"""End-to-end Pipecat voice pipeline: STT → LLM → Qwen3-TTS megakernel → audio.

Usage (requires OPENAI_API_KEY and DEEPGRAM_API_KEY env vars):

    python examples/pipecat_demo.py \
        --ref-audio /workspace/audio/reference.wav \
        --text "What is the speed of light?"

The --text flag runs in non-interactive text-in mode (no microphone needed).
Omit it for microphone input via Deepgram streaming STT.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import wave

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_megakernel.tts_stream import MegakernelQwen3TTS, TTSMetrics


async def run_text_mode(engine: MegakernelQwen3TTS, text: str, ref_audio: str, out: str) -> TTSMetrics:
    """Simple text-in → WAV-out path that exercises the full TTS stack."""
    metrics = TTSMetrics()
    sample_rate = int(engine.model.speech_tokenizer.get_output_sample_rate())
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        async for pcm, sr in engine.stream_pcm_chunks(
            text,
            language="English",
            ref_audio=ref_audio,
            x_vector_only_mode=True,
            code_chunk_steps=6,
            metrics=metrics,
        ):
            wf.writeframes(pcm)
            ttfc = f"{metrics.ttfc_ms:.1f} ms" if metrics.ttfc_ms else "pending"
            print(f"  chunk {metrics.chunks}: {len(pcm)//2} samples  TTFC={ttfc}")
    return metrics


async def run_pipecat_pipeline(engine: MegakernelQwen3TTS, ref_audio: str, text: str | None) -> None:
    """Full Pipecat STT→LLM→TTS pipeline.

    If text is provided, injects it directly (no microphone needed).
    Otherwise uses Deepgram streaming STT from the default mic.
    """
    try:
        from pipecat.frames.frames import LLMMessagesFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    except ImportError as e:
        print(f"Pipecat component not available: {e}")
        print("Install with: pip install pipecat-ai[openai,local]")
        return

    from qwen_megakernel.pipecat_service import QwenMegakernelTTSService

    tts = QwenMegakernelTTSService(
        engine=engine,
        ref_audio=ref_audio,
        language="English",
        code_chunk_steps=6,
    )

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=text is None,
            audio_out_enabled=True,
        )
    )

    llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
    context = OpenAILLMContext(
        messages=[{"role": "system", "content": "You are a helpful voice assistant. Keep answers concise (1-2 sentences)."}]
    )
    context_aggregator = llm.create_context_aggregator(context)

    pipeline_stages = [
        transport.input(),
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ]

    if text is None:
        try:
            from pipecat.services.deepgram.stt import DeepgramSTTService
            stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
            pipeline_stages.insert(1, stt)
        except ImportError:
            print("Deepgram STT not available; run with --text for TTS-only mode")
            return

    pipeline = Pipeline(pipeline_stages)
    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

    if text is not None:
        async def inject_text():
            await asyncio.sleep(0.1)
            await task.queue_frames([LLMMessagesFrame([{"role": "user", "content": text}])])

        asyncio.ensure_future(inject_text())

    runner = PipelineRunner()
    await runner.run(task)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--text", help="Text to synthesize (skips STT and LLM; outputs wav)")
    parser.add_argument("--out", default="demo_output.wav")
    parser.add_argument("--pipeline", action="store_true", help="Run full STT→LLM→TTS Pipecat pipeline")
    args = parser.parse_args()

    print(f"Loading model {args.model} ...")
    engine = MegakernelQwen3TTS.from_pretrained(args.model)
    print("Model loaded.")

    if args.pipeline or args.text and not args.pipeline:
        # Warm the voice cache before timing
        print("Warming voice cache ...")
        warmup_metrics = TTSMetrics()
        async for _ in engine.stream_pcm_chunks(
            "warm",
            ref_audio=args.ref_audio,
            x_vector_only_mode=True,
            code_chunk_steps=1,
            metrics=warmup_metrics,
        ):
            break

    if args.pipeline:
        print("Starting Pipecat pipeline ...")
        await run_pipecat_pipeline(engine, args.ref_audio, args.text)
    else:
        if not args.text:
            parser.error("Provide --text for TTS-only mode or --pipeline for full STT→LLM→TTS.")
        print(f"Synthesizing: {args.text!r}")
        metrics = await run_text_mode(engine, args.text, args.ref_audio, args.out)
        print(f"\nResults written to {args.out}")
        print(f"  TTFC:    {metrics.ttfc_ms:.1f} ms")
        print(f"  RTF:     {metrics.rtf:.3f}" if metrics.rtf else "  RTF:     n/a")
        print(f"  Audio:   {metrics.generated_audio_s:.2f} s")
        print(f"  Chunks:  {metrics.chunks}")
        print(f"  Prefill: {metrics.prefill_ms:.1f} ms")
        print(f"  CP kern: {metrics.code_predictor_ms:.1f} ms total")
        print(f"  Vocoder: {metrics.vocoder_ms:.1f} ms total")


if __name__ == "__main__":
    asyncio.run(main())
