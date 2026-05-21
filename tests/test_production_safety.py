"""
Production safety tests for MegakernelQwen3TTS.

Tests: ref-audio validation, concurrent-request lock, generation timeout,
error recovery (kernel reset on exception), and consumer abandonment.

No GPU or real model weights required — all CUDA/model code is mocked.
Run with:  python -m pytest tests/test_production_safety.py -v
       or: python tests/test_production_safety.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal torch stub so tts_stream.py can be imported without CUDA
# ---------------------------------------------------------------------------

class _Tensor:
    """Minimal tensor stand-in that handles all ops used inside stream_codebooks."""
    def __init__(self, val=None):
        self._val = val

    # shape mock: shape[1] returns 2 (one async prefill step + one sync step)
    @property
    def shape(self):
        return {1: 2}

    def to(self, *a, **kw):       return self
    def detach(self):             return self
    def cpu(self):                return self
    def contiguous(self):         return self
    def unsqueeze(self, *a):      return self
    def view(self, *a):           return self
    def expand(self, *a):         return self
    def sum(self, *a, **kw):      return self
    def chunk(self, n, **kw):     return [self] * n
    def __getitem__(self, key):   return self
    def __add__(self, other):     return self
    def __radd__(self, other):    return self


class _InferenceMode:
    """torch.inference_mode() used as a decorator — just passes the function through."""
    def __init__(self, *a, **kw): pass
    def __call__(self, fn):       return fn
    def __enter__(self):          return self
    def __exit__(self, *a):       pass


_torch = types.ModuleType("torch")
_torch.bfloat16     = "bfloat16"
_torch.long         = "long"
_torch.inference_mode = _InferenceMode
_torch.tensor       = lambda *a, **kw: _Tensor()
_torch.cat          = lambda tensors, **kw: _Tensor()
_torch.stack        = lambda tensors, **kw: _Tensor()
_torch.arange       = lambda *a, **kw: _Tensor()
sys.modules["torch"] = _torch

# Stub out numpy and the kernel module before importing tts_stream
_np = types.ModuleType("numpy")
_np.asarray  = lambda x, **kw: x
_np.clip     = lambda x, *a: x
_np.float32  = float
sys.modules["numpy"] = _np

_tts_kernel = types.ModuleType("qwen_megakernel.tts_kernel")
_tts_kernel.TTSTalkerMegakernel    = MagicMock
_tts_kernel.CodePredictorMegakernel = MagicMock
sys.modules["qwen_megakernel.tts_kernel"] = _tts_kernel

# Now import the real module under test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qwen_megakernel.tts_stream import MegakernelQwen3TTS  # noqa: E402

# ---------------------------------------------------------------------------
# Factory: build a stub engine with controllable kernel behaviour
# ---------------------------------------------------------------------------

EOS = 2150

def _make_engine(step_embed_side_effect=None, step_embed_return=EOS):
    """
    Build a MegakernelQwen3TTS instance with all GPU parts replaced by mocks.

    step_embed_side_effect: if set, kernel.step_embed raises this on the first call
    step_embed_return:      what kernel.step_embed returns on normal calls
    """
    model                                         = MagicMock()
    model.config.talker_config.codec_eos_token_id = EOS
    model.config.talker_config.codec_language_id  = {"english": 1}
    for attr in (
        "tts_bos_token_id", "tts_eos_token_id", "tts_pad_token_id",
        "talker_config.codec_nothink_id", "talker_config.codec_think_bos_id",
        "talker_config.codec_think_eos_id", "talker_config.codec_think_id",
        "talker_config.codec_pad_id", "talker_config.codec_bos_id",
    ):
        # each attribute resolves to a small distinct int via MagicMock auto-spec
        pass

    wrapper = MagicMock()
    wrapper.model = model

    engine                      = object.__new__(MegakernelQwen3TTS)
    engine.wrapper              = wrapper
    engine.model                = model
    engine.talker               = MagicMock()
    engine.talker.device        = "cpu"
    engine._voice_prompt_cache  = {}
    engine._inference_lock      = asyncio.Lock()
    engine._codec_embeds_stacked = _Tensor()
    engine._codec_embed_idx     = _Tensor()

    kernel = MagicMock()
    kernel.last_hidden_bf16 = _Tensor()
    if step_embed_side_effect is not None:
        kernel.step_embed.side_effect = step_embed_side_effect
    else:
        kernel.step_embed.return_value = step_embed_return
    engine.kernel = kernel

    engine.code_predictor_kernel = MagicMock()
    engine.code_predictor_kernel.generate.return_value = _Tensor()

    return engine


def _stub_prepare(engine):
    """Patch _prepare_voice_clone_inputs to return cheap fake tensors."""
    prompt_shape = {1: 2}  # 2 tokens → 1 async + 1 sync prefill step

    class FakePrompt(_Tensor):
        @property
        def shape(self): return prompt_shape

    def _fake_prepare(text, *, language, ref_audio, ref_text, x_vector_only_mode,
                      non_streaming_mode=False):
        t = FakePrompt()
        return t, t, t, 0.0

    engine._prepare_voice_clone_inputs = _fake_prepare


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRefAudioValidation(unittest.TestCase):

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            MegakernelQwen3TTS._validate_ref_audio("/nonexistent/path/ref.wav")

    def test_existing_file_passes(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            MegakernelQwen3TTS._validate_ref_audio(path)  # must not raise
        finally:
            os.unlink(path)

    def test_non_path_value_passes(self):
        # Passing a bytes or array object (not a path) is allowed
        MegakernelQwen3TTS._validate_ref_audio(b"\x00\x01\x02")

    def test_warm_speaker_raises_on_missing_file(self):
        engine = _make_engine()
        with self.assertRaises(FileNotFoundError):
            engine.warm_speaker("/does/not/exist.wav")

    def test_stream_codebooks_raises_on_missing_file(self):
        engine = _make_engine()
        _stub_prepare(engine)

        async def run():
            async for _ in engine.stream_codebooks("hi", ref_audio="/does/not/exist.wav"):
                pass

        with self.assertRaises(FileNotFoundError):
            asyncio.run(run())


class TestConcurrentRequestLock(unittest.TestCase):

    def test_two_requests_are_serialised(self):
        """
        Launch two stream_codebooks coroutines concurrently.
        The lock must prevent them from overlapping — they should
        complete in sequence, not simultaneously.
        """
        order = []

        engine = _make_engine(step_embed_return=EOS)
        _stub_prepare(engine)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref = f.name

        try:
            async def run_one(label):
                order.append(f"{label}:start")
                async for _ in engine.stream_codebooks("hi", ref_audio=ref):
                    pass
                order.append(f"{label}:end")

            async def run_both():
                await asyncio.gather(run_one("A"), run_one("B"))

            asyncio.run(run_both())
        finally:
            os.unlink(ref)

        # With the lock, one request finishes entirely before the other starts
        self.assertEqual(order[1], "A:end",
            f"Request A should finish before B starts. Got: {order}")
        self.assertEqual(order[2], "B:start",
            f"Request B should start after A finishes. Got: {order}")

    def test_lock_is_released_after_normal_completion(self):
        engine = _make_engine(step_embed_return=EOS)
        _stub_prepare(engine)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref = f.name

        try:
            async def run():
                async for _ in engine.stream_codebooks("hi", ref_audio=ref):
                    pass
                # Lock must be free after the generator is exhausted
                self.assertFalse(engine._inference_lock.locked())

            asyncio.run(run())
        finally:
            os.unlink(ref)


class TestGenerationTimeout(unittest.TestCase):

    def test_timeout_raises_after_deadline(self):
        """
        Make step_embed always return a non-EOS token so generation runs
        forever — the timeout must fire and raise asyncio.TimeoutError.
        """
        NON_EOS = 1000
        engine  = _make_engine(step_embed_return=NON_EOS)
        _stub_prepare(engine)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref = f.name

        try:
            async def run():
                async for _ in engine.stream_codebooks(
                    "hi", ref_audio=ref,
                    generation_timeout_s=0.05,   # 50 ms — fires almost immediately
                    max_repeat=9999,             # disable repeat-stop so timeout fires first
                ):
                    pass

            with self.assertRaises(asyncio.TimeoutError):
                asyncio.run(run())
        finally:
            os.unlink(ref)

    def test_timeout_releases_lock(self):
        NON_EOS = 1000
        engine  = _make_engine(step_embed_return=NON_EOS)
        _stub_prepare(engine)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref = f.name

        try:
            async def run():
                try:
                    async for _ in engine.stream_codebooks(
                        "hi", ref_audio=ref,
                        generation_timeout_s=0.05,
                        max_repeat=9999,
                    ):
                        pass
                except asyncio.TimeoutError:
                    pass
                self.assertFalse(engine._inference_lock.locked())

            asyncio.run(run())
        finally:
            os.unlink(ref)


class TestErrorRecovery(unittest.TestCase):

    def test_gpu_error_resets_kernel_and_reraises(self):
        """
        If step_embed raises (e.g. GPU OOM), kernel.reset() must be called
        and the exception must propagate to the caller.
        """
        engine = _make_engine(step_embed_side_effect=RuntimeError("CUDA OOM"))
        _stub_prepare(engine)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref = f.name

        try:
            async def run():
                async for _ in engine.stream_codebooks("hi", ref_audio=ref):
                    pass

            with self.assertRaises(RuntimeError, msg="Exception must propagate"):
                asyncio.run(run())

            # kernel.reset() must have been called at least once during error cleanup
            reset_calls = engine.kernel.reset.call_count
            self.assertGreaterEqual(reset_calls, 1,
                f"kernel.reset() should be called on error, got {reset_calls} calls")
        finally:
            os.unlink(ref)

    def test_lock_released_after_error(self):
        engine = _make_engine(step_embed_side_effect=RuntimeError("boom"))
        _stub_prepare(engine)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref = f.name

        try:
            async def run():
                try:
                    async for _ in engine.stream_codebooks("hi", ref_audio=ref):
                        pass
                except RuntimeError:
                    pass
                self.assertFalse(engine._inference_lock.locked(),
                    "Lock must be released even after an error")

            asyncio.run(run())
        finally:
            os.unlink(ref)

    def test_second_request_succeeds_after_error(self):
        """After a failed request, the next one must start with a clean kernel."""
        call_count = {"n": 0}
        EOS_VAL = EOS

        def step_side_effect(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first call fails")
            return EOS_VAL

        engine = _make_engine(step_embed_side_effect=step_side_effect)
        _stub_prepare(engine)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref = f.name

        try:
            async def run():
                # First request — fails
                try:
                    async for _ in engine.stream_codebooks("hi", ref_audio=ref):
                        pass
                except RuntimeError:
                    pass

                # Second request — must succeed (kernel was reset)
                chunks = []
                async for code in engine.stream_codebooks("hi", ref_audio=ref):
                    chunks.append(code)
                # EOS returned immediately → 0 codebook chunks emitted, no crash

            asyncio.run(run())  # must not raise
        finally:
            os.unlink(ref)


class TestConsumerAbandonment(unittest.TestCase):

    def test_lock_released_when_consumer_breaks_early(self):
        """
        If the caller breaks out of the async for loop mid-stream,
        the lock must still be released so the next request can proceed.
        """
        NON_EOS = 1000
        engine  = _make_engine(step_embed_return=NON_EOS)
        _stub_prepare(engine)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref = f.name

        try:
            async def run():
                gen = engine.stream_codebooks(
                    "hi", ref_audio=ref,
                    max_repeat=9999,
                    generation_timeout_s=5.0,
                )
                async for _ in gen:
                    break  # abandon after first chunk
                await gen.aclose()

                self.assertFalse(engine._inference_lock.locked(),
                    "Lock must be released after consumer abandons the generator")

            asyncio.run(run())
        finally:
            os.unlink(ref)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
