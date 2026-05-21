"""Qwen Megakernel - single-kernel Qwen3 decode for RTX 5090."""

__all__ = ["load_weights", "Decoder", "generate"]


def __getattr__(name):
    if name in __all__:
        from qwen_megakernel import model

        return getattr(model, name)
    raise AttributeError(name)
