"""Correctness verification: megakernel talker vs reference Qwen3-TTS.

The megakernel replaces Qwen3-TTS's 28-layer talker forward pass. CHANGES.md
Step 8 notes the megakernel diverges from the reference (gets stuck repeating a
token, never emits EOS) and is stopped via a `max_repeat` heuristic. Nothing in
the repo verifies whether the *audio* is actually correct on real sentences --
the only thing ever tested is the single word "Hello".

This script closes that gap. For each sentence it:

  1. Generates audio with the REFERENCE path  (model.generate_voice_clone)
  2. Generates audio with the MEGAKERNEL path  (engine.stream_pcm_chunks)
  3. Transcribes both with Whisper
  4. Reports WER(reference_transcript vs input), WER(megakernel vs input),
     and WER(megakernel vs reference) -- plus duration / RMS sanity stats.

WER is the right first-order signal: it is robust to the megakernel's
silence-padded tail and to prosody/duration differences, and it directly
answers "do intelligible, correct words come out?". If megakernel-vs-input WER
tracks reference-vs-input WER across sentence lengths, the kernel is sound. If
it blows up as sentences get longer, the divergence is a real bug, not benign
bf16 drift, and `max_repeat` is hiding it.

USAGE (on an RTX 5090 with qwen_tts + the model installed):

    pip install -U openai-whisper            # or: pip install faster-whisper
    python examples/verify_correctness.py \
        --ref-audio /path/to/clone.wav \
        --ref-text "transcript of the reference clip" \
        --out-dir ./verify_out

Add --texts-file sentences.txt (one sentence per line) to use your own set.

NOTE: the reference call below uses `model.generate_voice_clone(...)`, which is
the documented public API. If the installed qwen_tts version names arguments
differently, adjust `_reference_wav()` -- it is the only API-coupled piece.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import re
import wave
from pathlib import Path

import numpy as np
import torch

from qwen_megakernel.tts_stream import MegakernelQwen3TTS, TTSMetrics

# Varied lengths on purpose: divergence bugs show up as sentences get longer,
# so a single short utterance ("Hello") cannot reveal them.
DEFAULT_TEXTS = [
    "Hello.",
    "The quick brown fox jumps over the lazy dog.",
    "Real time speech synthesis on a single graphics card is finally practical.",
    "She sells sea shells by the sea shore, and the shells she sells are surely "
    "sea shells, so if she sells shells on the seashore, the shells are seashore shells.",
    "In nineteen sixty nine, the first humans landed on the moon, and the whole "
    "world watched as a new era of exploration began, changing how we see ourselves.",
]


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #

def _save_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)


def _pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767.0


def _audio_stats(wav: np.ndarray, sr: int) -> dict:
    wav = np.asarray(wav, dtype=np.float32)
    return {
        "duration_s": round(len(wav) / sr, 3),
        "rms": round(float(np.sqrt(np.mean(wav**2) + 1e-12)), 5),
        "peak": round(float(np.max(np.abs(wav)) if len(wav) else 0.0), 5),
    }


# --------------------------------------------------------------------------- #
# WER
# --------------------------------------------------------------------------- #

def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return text.split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate via Levenshtein distance over word tokens."""
    r, h = _normalize(reference), _normalize(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    # DP edit distance
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cost = 0 if rw == hw else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(h)] / len(r)


# --------------------------------------------------------------------------- #
# Whisper (lazy, supports faster-whisper or openai-whisper)
# --------------------------------------------------------------------------- #

class Transcriber:
    def __init__(self, model_size: str = "base.en"):
        self._backend = None
        self._model = None
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(model_size, device="cuda", compute_type="float16")
            self._backend = "faster"
            return
        except Exception:
            pass
        try:
            import whisper

            self._model = whisper.load_model(model_size.replace(".en", "") if "large" in model_size else model_size)
            self._backend = "openai"
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "No Whisper backend found. Install one of:\n"
                "  pip install faster-whisper   (recommended, GPU)\n"
                "  pip install -U openai-whisper"
            ) from exc

    def transcribe(self, wav_path: Path) -> str:
        if self._backend == "faster":
            segments, _ = self._model.transcribe(str(wav_path), language="en")
            return " ".join(s.text for s in segments).strip()
        result = self._model.transcribe(str(wav_path), language="en")
        return str(result.get("text", "")).strip()


# --------------------------------------------------------------------------- #
# Generation paths
# --------------------------------------------------------------------------- #

def _reference_wav(engine, text, *, language, ref_audio, ref_text):
    """Reference (no-megakernel) path via the official public API.

    engine.wrapper IS the loaded Qwen3TTSModel, so this reuses the already-loaded
    weights -- no second download. This is the ONLY API-coupled function; adjust
    here if the installed qwen_tts signature differs.
    """
    wavs, sr = engine.wrapper.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    wav = wavs[0]
    if hasattr(wav, "detach"):
        wav = wav.detach().to("cpu", torch.float32).numpy()
    return np.asarray(wav, dtype=np.float32), int(sr)


async def _megakernel_wav(engine, text, *, language, ref_audio, ref_text,
                          x_vector_only_mode, chunk_steps):
    metrics = TTSMetrics()
    chunks: list[bytes] = []
    sr = 24000
    async for pcm, s in engine.stream_pcm_chunks(
        text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only_mode,
        code_chunk_steps=chunk_steps,
        metrics=metrics,
    ):
        chunks.append(pcm)
        sr = s
    wav = _pcm16_to_float(b"".join(chunks)) if chunks else np.zeros(1, dtype=np.float32)
    return wav, sr, metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--ref-text", help="Transcript of the reference clip. Required for a fair ICL comparison.")
    ap.add_argument("--language", default="English")
    ap.add_argument("--texts-file", help="One sentence per line. Defaults to a built-in varied-length set.")
    ap.add_argument("--out-dir", default="./verify_out")
    ap.add_argument("--whisper-model", default="base.en")
    ap.add_argument("--chunk-steps", type=int, default=6)
    ap.add_argument("--x-vector-only", action="store_true",
                    help="Megakernel speaker-vector-only mode. Off by default so it matches "
                         "generate_voice_clone's ICL conditioning for a fair comparison.")
    ap.add_argument("--max-fail-wer", type=float, default=0.5,
                    help="Flag a sentence if megakernel WER exceeds reference WER by more than this.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = (
        [ln.strip() for ln in Path(args.texts_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if args.texts_file
        else DEFAULT_TEXTS
    )

    print(f"Loading {args.model} ...")
    engine = MegakernelQwen3TTS.from_pretrained(args.model)
    # Exclude the one-time speaker-encode cost from the megakernel timing.
    engine.warm_speaker(args.ref_audio, ref_text=args.ref_text,
                        x_vector_only_mode=args.x_vector_only)

    print(f"Loading Whisper ({args.whisper_model}) ...")
    asr = Transcriber(args.whisper_model)

    rows = []
    for i, text in enumerate(texts):
        print(f"\n[{i+1}/{len(texts)}] {text!r}")

        ref_wav, ref_sr = _reference_wav(
            engine, text, language=args.language,
            ref_audio=args.ref_audio, ref_text=args.ref_text,
        )
        mk_wav, mk_sr, metrics = await _megakernel_wav(
            engine, text, language=args.language,
            ref_audio=args.ref_audio, ref_text=args.ref_text,
            x_vector_only_mode=args.x_vector_only, chunk_steps=args.chunk_steps,
        )

        ref_path = out_dir / f"sent{i:02d}_reference.wav"
        mk_path = out_dir / f"sent{i:02d}_megakernel.wav"
        _save_wav(ref_path, ref_wav, ref_sr)
        _save_wav(mk_path, mk_wav, mk_sr)

        ref_tx = asr.transcribe(ref_path)
        mk_tx = asr.transcribe(mk_path)

        row = {
            "idx": i,
            "text": text,
            "ref_transcript": ref_tx,
            "mk_transcript": mk_tx,
            "wer_ref_vs_input": round(wer(text, ref_tx), 3),
            "wer_mk_vs_input": round(wer(text, mk_tx), 3),
            "wer_mk_vs_ref": round(wer(ref_tx, mk_tx), 3),
            "ref_audio": _audio_stats(ref_wav, ref_sr),
            "mk_audio": _audio_stats(mk_wav, mk_sr),
            "ttfc_ms": round(metrics.ttfc_ms, 1) if metrics.ttfc_ms else None,
            "rtf": round(metrics.rtf, 4) if metrics.rtf else None,
            "codec_steps": metrics.codec_steps,
        }
        # A sentence "fails" if the megakernel is much worse than the reference,
        # i.e. the kernel -- not the model -- introduced the errors.
        row["FLAG"] = row["wer_mk_vs_input"] - row["wer_ref_vs_input"] > args.max_fail_wer
        rows.append(row)

        print(f"  ref  : {ref_tx!r}  (WER {row['wer_ref_vs_input']})")
        print(f"  mk   : {mk_tx!r}  (WER {row['wer_mk_vs_input']}, vs ref {row['wer_mk_vs_ref']})")
        print(f"  ttfc={row['ttfc_ms']}ms rtf={row['rtf']} steps={metrics.codec_steps}"
              f"{'   <<< FLAG: kernel-induced errors' if row['FLAG'] else ''}")

    (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    n = len(rows)
    flagged = [r for r in rows if r["FLAG"]]
    avg_ref = sum(r["wer_ref_vs_input"] for r in rows) / n
    avg_mk = sum(r["wer_mk_vs_input"] for r in rows) / n
    print("\n" + "=" * 64)
    print("SUMMARY")
    print(f"  sentences            : {n}")
    print(f"  avg WER ref vs input : {avg_ref:.3f}   (model's own floor)")
    print(f"  avg WER mk  vs input : {avg_mk:.3f}   (megakernel)")
    print(f"  WER gap (mk - ref)   : {avg_mk - avg_ref:+.3f}")
    print(f"  flagged sentences    : {len(flagged)}  {[r['idx'] for r in flagged]}")
    print(f"  details              : {out_dir / 'results.json'}")
    if avg_mk - avg_ref > 0.1:
        print("\n  VERDICT: megakernel WER is materially worse than the reference -> likely a")
        print("           real kernel divergence bug, not benign bf16 drift. Investigate RoPE")
        print("           theta/convention, qk-norm, attn_scale before trusting the benchmarks.")
    else:
        print("\n  VERDICT: megakernel tracks the reference -> divergence is benign tail drift.")
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(main())
