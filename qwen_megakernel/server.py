"""Simple streaming inference server for the Qwen3-TTS megakernel backend."""

from __future__ import annotations

import base64
import json
import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from qwen_megakernel.tts_stream import MegakernelQwen3TTS, TTSMetrics


class TTSRequest(BaseModel):
    text: str
    language: str = "English"
    ref_audio: str
    ref_text: str | None = None
    x_vector_only_mode: bool = True
    code_chunk_steps: int = 6
    log_path: str | None = None


app = FastAPI(title="Qwen3-TTS Megakernel")
_engine: MegakernelQwen3TTS | None = None


def engine() -> MegakernelQwen3TTS:
    global _engine
    if _engine is None:
        _engine = MegakernelQwen3TTS.from_pretrained(
            os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        )
    return _engine


@app.post("/stream_audio")
async def stream_audio(req: TTSRequest):
    metrics = TTSMetrics()

    async def body():
        async for pcm, sr in engine().stream_pcm_chunks(
            req.text,
            language=req.language,
            ref_audio=req.ref_audio,
            ref_text=req.ref_text,
            x_vector_only_mode=req.x_vector_only_mode,
            code_chunk_steps=req.code_chunk_steps,
            metrics=metrics,
            log_path=req.log_path,
        ):
            yield (
                json.dumps(
                    {
                        "type": "audio",
                        "sample_rate": sr,
                        "encoding": "pcm_s16le",
                        "audio_b64": base64.b64encode(pcm).decode("ascii"),
                    }
                )
                + "\n"
            )
        yield json.dumps({"type": "metrics", **metrics.__dict__}) + "\n"

    return StreamingResponse(body(), media_type="application/x-ndjson")


@app.post("/stream_codes")
async def stream_codes(req: TTSRequest):
    metrics = TTSMetrics()

    async def body():
        async for codes in engine().stream_codebooks(
            req.text,
            language=req.language,
            ref_audio=req.ref_audio,
            ref_text=req.ref_text,
            x_vector_only_mode=req.x_vector_only_mode,
            metrics=metrics,
        ):
            yield json.dumps({"type": "codes", "codes": codes.cpu().tolist()}) + "\n"
        yield json.dumps({"type": "metrics", **metrics.__dict__}) + "\n"

    return StreamingResponse(body(), media_type="application/x-ndjson")
