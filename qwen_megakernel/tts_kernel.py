"""Qwen3-TTS talker decode helpers backed by the megakernel.

This module intentionally accelerates only the talker transformer that predicts
the first codec codebook. Qwen3-TTS's smaller code predictor and speech tokenizer
stay in the official PyTorch implementation.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass

import torch

# Importing qwen_megakernel.model would build the extension with the text-vocab
# default. The TTS talker head is 3072 rows, so set the compile-time override
# before touching torch.ops.
os.environ.setdefault("LDG_VOCAB_SIZE", "3072")
# With vocab=3072, LM head only needs ceil(3072/24)=128 blocks (not the text default of 1280).
# Fewer blocks = less wasted SM occupancy on the argmax kernel.
os.environ.setdefault("LDG_LM_NUM_BLOCKS", "128")

from qwen_megakernel.build import get_extension as _get_ext  # noqa: E402

_get_ext()

_decode_embed = torch.ops.qwen_megakernel_C.decode_embed


@dataclass(frozen=True)
class TalkerShape:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    max_seq_len: int
    rope_theta: float

    @property
    def q_size(self) -> int:
        return 16 * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim


def _pack_layer_weights(layer_weights: list[torch.Tensor], num_layers: int) -> torch.Tensor:
    ptr_size = 8
    n_ptrs = 11
    buf = bytearray(num_layers * n_ptrs * ptr_size)
    for i in range(num_layers):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


def _rope_tables(shape: TalkerShape) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        shape.rope_theta
        ** (torch.arange(0, shape.head_dim, 2, dtype=torch.float32) / shape.head_dim)
    )
    positions = torch.arange(shape.max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    return cos, sin


def _contiguous_weight(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cuda", dtype=torch.bfloat16).contiguous()


def extract_talker_weights(talker) -> tuple[dict[str, object], TalkerShape]:
    """Extract weight tensors from Qwen3TTSTalkerForConditionalGeneration."""
    cfg = talker.config
    if cfg.hidden_size != 1024 or cfg.intermediate_size != 3072:
        raise ValueError(
            "This megakernel path is specialized for the 0.6B talker shape "
            f"(got hidden={cfg.hidden_size}, intermediate={cfg.intermediate_size})."
        )
    if cfg.vocab_size != 3072:
        raise ValueError(f"Expected Qwen3-TTS 12Hz talker vocab 3072, got {cfg.vocab_size}.")

    shape = TalkerShape(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        max_seq_len=min(getattr(cfg, "max_position_embeddings", 4096), int(os.getenv("LDG_MAX_SEQ_LEN", "4096"))),
        rope_theta=float(getattr(cfg, "rope_theta", 1_000_000.0)),
    )

    layer_weights: list[torch.Tensor] = []
    for layer in talker.model.layers:
        layer_weights.extend(
            [
                _contiguous_weight(layer.input_layernorm.weight),
                _contiguous_weight(layer.self_attn.q_proj.weight),
                _contiguous_weight(layer.self_attn.k_proj.weight),
                _contiguous_weight(layer.self_attn.v_proj.weight),
                _contiguous_weight(layer.self_attn.q_norm.weight),
                _contiguous_weight(layer.self_attn.k_norm.weight),
                _contiguous_weight(layer.self_attn.o_proj.weight),
                _contiguous_weight(layer.post_attention_layernorm.weight),
                _contiguous_weight(layer.mlp.gate_proj.weight),
                _contiguous_weight(layer.mlp.up_proj.weight),
                _contiguous_weight(layer.mlp.down_proj.weight),
            ]
        )

    weights = {
        "layer_weights": layer_weights,
        "final_norm_weight": _contiguous_weight(talker.model.norm.weight),
        "codec_head_weight": _contiguous_weight(talker.codec_head.weight),
    }
    return weights, shape


def extract_code_predictor_weights(
    code_predictor, max_seq_len: int = 32
) -> tuple[dict[str, object], "TalkerShape"]:
    """Extract weights from the Qwen3-TTS code_predictor for megakernel use.

    The code_predictor is a 5-layer Qwen3 backbone with 15 per-codebook lm_heads
    and 15 embedding tables (one per codebook group 2-16).  Its weight shapes
    are identical to the talker's, so the same compiled kernel handles it with
    num_layers=5.  The lm_heads are zero-padded from their actual codebook vocab
    (typically 2048) to LDG_VOCAB_SIZE=3072 so the argmax kernel is correct.
    """
    model = code_predictor.model
    cfg = code_predictor.config

    num_layers = len(list(model.layers))
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // getattr(cfg, "num_attention_heads", 16))
    shape = TalkerShape(
        num_layers=num_layers,
        num_kv_heads=getattr(cfg, "num_key_value_heads", 8),
        head_dim=head_dim,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        vocab_size=3072,
        max_seq_len=max_seq_len,
        rope_theta=float(getattr(cfg, "rope_theta", 1_000_000.0)),
    )

    layer_weights: list[torch.Tensor] = []
    for layer in model.layers:
        layer_weights.extend([
            _contiguous_weight(layer.input_layernorm.weight),
            _contiguous_weight(layer.self_attn.q_proj.weight),
            _contiguous_weight(layer.self_attn.k_proj.weight),
            _contiguous_weight(layer.self_attn.v_proj.weight),
            _contiguous_weight(layer.self_attn.q_norm.weight),
            _contiguous_weight(layer.self_attn.k_norm.weight),
            _contiguous_weight(layer.self_attn.o_proj.weight),
            _contiguous_weight(layer.post_attention_layernorm.weight),
            _contiguous_weight(layer.mlp.gate_proj.weight),
            _contiguous_weight(layer.mlp.up_proj.weight),
            _contiguous_weight(layer.mlp.down_proj.weight),
        ])

    # Try both singular (actual Qwen3-TTS name) and plural fallbacks
    for _attr in ("lm_head", "lm_heads", "codec_heads"):
        _candidate = getattr(code_predictor, _attr, None)
        if _candidate is not None and hasattr(_candidate, "__len__") and len(_candidate) > 1:
            lm_heads_obj = _candidate
            break
    else:
        raise AttributeError(
            f"Cannot find a ModuleList of lm_heads on {type(code_predictor).__name__}. "
            f"Named children: {[n for n, _ in code_predictor.named_children()]}"
        )
    lm_heads_padded: list[torch.Tensor] = []
    for head in lm_heads_obj:
        w = _contiguous_weight(head.weight)
        if w.shape[0] > 3072:
            raise ValueError(
                f"code_predictor lm_head vocab {w.shape[0]} > LDG_VOCAB_SIZE=3072; "
                "zero-padding trick requires lm_head vocab ≤ LDG_VOCAB_SIZE."
            )
        if w.shape[0] < 3072:
            pad = torch.zeros(3072 - w.shape[0], w.shape[1], dtype=torch.bfloat16, device="cuda")
            w = torch.cat([w, pad], dim=0).contiguous()
        lm_heads_padded.append(w)

    raw_embeds = code_predictor.get_input_embeddings()
    n_heads = len(lm_heads_padded)
    try:
        raw_embed_weights = [raw_embeds[i].weight for i in range(n_heads)]
    except (TypeError, KeyError) as exc:
        raise TypeError(
            f"code_predictor.get_input_embeddings() must be indexable; got {type(raw_embeds)}"
        ) from exc

    # Actual codebook vocab size (before padding) — used for output clamping
    codebook_vocab_size: int = raw_embed_weights[0].shape[0]

    # Pad embeddings to 3072 (same as lm_heads) so tokens in the padded range
    # [codebook_vocab_size, 3071] never cause an index-out-of-bounds lookup when
    # the zero-padded lm_head rows win the argmax (all real logits negative edge case).
    embeddings_padded: list[torch.Tensor] = []
    for w in raw_embed_weights:
        w = _contiguous_weight(w)
        if w.shape[0] < 3072:
            pad = torch.zeros(3072 - w.shape[0], w.shape[1], dtype=torch.bfloat16, device="cuda")
            w = torch.cat([w, pad], dim=0).contiguous()
        embeddings_padded.append(w)

    return {
        "layer_weights": layer_weights,
        "final_norm_weight": _contiguous_weight(model.norm.weight),
        "lm_heads_padded": lm_heads_padded,
        "embeddings": embeddings_padded,
        "codebook_vocab_size": codebook_vocab_size,
    }, shape


class CodePredictorMegakernel:
    """5-layer code_predictor backbone driven by the megakernel.

    Replaces code_predictor.generate() (~94 ms/step) with an async GPU AR loop
    (~1.2 ms for all 15 steps, zero CPU syncs) on RTX 5090.
    """

    NUM_CODEBOOKS = 15  # codebook groups 2-16

    def __init__(self, code_predictor, max_seq_len: int = 32):
        weights, shape = extract_code_predictor_weights(code_predictor, max_seq_len=max_seq_len)
        self.shape = shape
        self._weights = weights
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"], shape.num_layers)
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_heads_padded: list[torch.Tensor] = weights["lm_heads_padded"]
        self._embeddings: list[torch.Tensor] = weights["embeddings"]
        self._codebook_vocab_size: int = weights["codebook_vocab_size"]
        self._cos_table, self._sin_table = _rope_tables(shape)
        self._attn_scale = 1.0 / math.sqrt(shape.head_dim)

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        self._k_cache = torch.zeros(
            shape.num_layers, shape.num_kv_heads, shape.max_seq_len, shape.head_dim, **bf16
        )
        self._v_cache = torch.zeros_like(self._k_cache)
        self._hidden = torch.empty(shape.hidden_size, **bf16)
        self._act = torch.empty(shape.hidden_size, **f32)
        self._res = torch.empty(shape.hidden_size, **f32)
        self._q = torch.empty(shape.q_size, **f32)
        self._k = torch.empty(shape.kv_size, **f32)
        self._v = torch.empty(shape.kv_size, **f32)
        self._attn_out = torch.empty(shape.q_size, **f32)
        self._mlp_inter = torch.empty(shape.intermediate_size, **f32)
        self._norm_out = torch.empty(shape.hidden_size, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")
        # int64 mirror of _out_token for GPU embedding lookups (avoids CPU sync)
        self._out_token_long = torch.empty(1, dtype=torch.int64, device="cuda")
        # Accumulation buffer: collect all NUM_CODEBOOKS tokens without CPU sync
        self._token_buf = torch.zeros(self.NUM_CODEBOOKS, dtype=torch.int32, device="cuda")
        self._position = 0

    def _reset(self) -> None:
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    def _step_async(self, embed: torch.Tensor, lm_head_idx: int) -> None:
        """Run one decode step.  Does NOT call .item() — stays fully on GPU.

        Writes the argmax token to self._out_token (int32) and its int64 copy
        to self._out_token_long.  Both are ready for the next GPU op on the same
        CUDA stream without any CPU–GPU sync.
        """
        if self._position >= self.shape.max_seq_len:
            raise RuntimeError(f"CodePredictorMegakernel KV cache exhausted at {self._position}.")
        if embed.ndim == 3:
            embed = embed[0, 0]
        elif embed.ndim == 2:
            embed = embed[0]
        embed = embed.to(device="cuda", dtype=torch.bfloat16).contiguous()
        _decode_embed(
            self._out_token,
            embed,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_heads_padded[lm_head_idx],
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            self.shape.num_layers,
            self._position,
            self.shape.max_seq_len,
            self._attn_scale,
        )
        # Async int32→int64 copy so embedding lookups can use _out_token_long
        # directly on the GPU without a CPU read.  CUDA stream order guarantees
        # this copy sees the token written by _decode_embed above.
        self._out_token_long.copy_(self._out_token)
        self._position += 1

    def generate(
        self, past_hidden: torch.Tensor, last_id_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Generate 15 codebook tokens from talker context.

        The entire AR loop runs without CPU–GPU syncs.  All embedding lookups
        use torch.index_select with the GPU _out_token_long buffer so CUDA stream
        serialisation guarantees correctness.  A single forced sync occurs later
        when the caller reads the returned tensor (via .cpu() in stream_pcm_chunks).

        Args:
            past_hidden:    [1, 1, 1024] bf16 — last talker hidden state
            last_id_hidden: [1, 1, 1024] bf16 — embedding of first codebook token

        Returns:
            [1, 15] int64 CUDA tensor — codebook tokens 2-16.
        """
        self._reset()

        # Prefill position 0 with past_hidden; output is not stored
        self._step_async(past_hidden, 0)

        # Position 1: last_id_hidden → token[0] via lm_head[0]
        self._step_async(last_id_hidden, 0)
        self._token_buf[0:1].copy_(self._out_token)

        # Positions 2-15: fully GPU AR loop, zero CPU syncs
        for i in range(1, self.NUM_CODEBOOKS):
            # index_select is async: CUDA stream ensures _out_token_long holds
            # the token from the previous _step_async before this runs.
            emb = torch.index_select(
                self._embeddings[i - 1], 0, self._out_token_long
            ).view(self.shape.hidden_size)
            self._step_async(emb, i)
            self._token_buf[i : i + 1].copy_(self._out_token)

        # Clamp: zero-padded lm_head rows (>= codebook_vocab_size) should not
        # be returned as real tokens — clamp to valid vocab range.
        return self._token_buf.long().unsqueeze(0).clamp_(0, self._codebook_vocab_size - 1)


class TTSTalkerMegakernel:
    """Stateful embedding-in decoder for Qwen3-TTS's talker transformer."""

    def __init__(self, talker, max_seq_len: int | None = None):
        weights, shape = extract_talker_weights(talker)
        if max_seq_len is not None:
            shape = TalkerShape(**{**shape.__dict__, "max_seq_len": max_seq_len})
        self.shape = shape
        self._position = 0
        self._weights = weights
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"], shape.num_layers)
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["codec_head_weight"]
        self._cos_table, self._sin_table = _rope_tables(shape)
        self._attn_scale = 1.0 / math.sqrt(shape.head_dim)

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        self._k_cache = torch.zeros(
            shape.num_layers,
            shape.num_kv_heads,
            shape.max_seq_len,
            shape.head_dim,
            **bf16,
        )
        self._v_cache = torch.zeros_like(self._k_cache)
        self._hidden = torch.empty(shape.hidden_size, **bf16)
        self._act = torch.empty(shape.hidden_size, **f32)
        self._res = torch.empty(shape.hidden_size, **f32)
        self._q = torch.empty(shape.q_size, **f32)
        self._k = torch.empty(shape.kv_size, **f32)
        self._v = torch.empty(shape.kv_size, **f32)
        self._attn_out = torch.empty(shape.q_size, **f32)
        self._mlp_inter = torch.empty(shape.intermediate_size, **f32)
        self._norm_out = torch.empty(shape.hidden_size, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

    def reset(self) -> None:
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    def step_embed(self, input_embed: torch.Tensor) -> int:
        if self._position >= self.shape.max_seq_len:
            raise RuntimeError(f"KV cache exhausted at {self._position} tokens.")
        if input_embed.ndim == 3:
            input_embed = input_embed[0, 0]
        elif input_embed.ndim == 2:
            input_embed = input_embed[0]
        input_embed = input_embed.to(device="cuda", dtype=torch.bfloat16).contiguous()
        _decode_embed(
            self._out_token,
            input_embed,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            self.shape.num_layers,
            self._position,
            self.shape.max_seq_len,
            self._attn_scale,
        )
        self._position += 1
        return int(self._out_token.item())

    def step_embed_async(self, input_embed: torch.Tensor) -> None:
        """Like step_embed but skips .item() sync — output stays in self._out_token on GPU."""
        if self._position >= self.shape.max_seq_len:
            raise RuntimeError(f"KV cache exhausted at {self._position} tokens.")
        if input_embed.ndim == 3:
            input_embed = input_embed[0, 0]
        elif input_embed.ndim == 2:
            input_embed = input_embed[0]
        input_embed = input_embed.to(device="cuda", dtype=torch.bfloat16).contiguous()
        _decode_embed(
            self._out_token,
            input_embed,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            self.shape.num_layers,
            self._position,
            self.shape.max_seq_len,
            self._attn_scale,
        )
        self._position += 1

    @property
    def last_hidden_bf16(self) -> torch.Tensor:
        return self._norm_out.to(torch.bfloat16).view(1, 1, -1)

    @property
    def position(self) -> int:
        return self._position
