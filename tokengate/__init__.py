"""TokenGate: end-to-end learnable KV-cache token pruning."""

_TRAIN_API = ("TokenGateConfig", "TokenGateTrainer")
_INFERENCE_API = ("load_model", "TokenGateModel", "GenerateOutput",
                  "apply_chat_template", "split_state_span",
                  "STATE_START_ID", "STATE_END_ID")
__all__ = [*_TRAIN_API, *_INFERENCE_API]


def __getattr__(name):
    # Lazy: `import tokengate` must stay light (and must not pull in torch).
    if name in _TRAIN_API:
        from tokengate import trainer
        return getattr(trainer, name)
    if name in _INFERENCE_API:
        from tokengate import inference
        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
