"""Model and training configuration for OpenMythos.

A configuration is a plain dataclass so that every experiment can be fully
described by a JSON-serialisable object. Two presets ship with the project:

- ``tiny_debug``: ~1M params, runs forward/backward/generation on CPU in seconds.
- ``mini_60m``:   ~62M params, the reference model sized for a single T4 (16GB).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # Vocabulary / sequence. Byte-level tokenization: ids 0..255 are raw bytes,
    # 256..259 are specials. vocab_size is padded to a multiple of 64.
    vocab_size: int = 512
    max_seq_len: int = 2048

    # Core dimensions (mini_60m preset: d_model=2048, d_ff=8192 -> ~62M params)
    d_model: int = 2048
    n_heads: int = 32          # query heads
    n_kv_heads: int = 8        # key/value heads (grouped-query attention)
    d_ff: int = 8192           # SwiGLU intermediate size

    # Recurrence (Prelude -> RecurrentBlock x N -> Coda). The block weights are
    # *shared* across iterations: depth grows without growing the parameter count.
    n_recurrent: int = 8

    # Normalisation / activation
    norm_eps: float = 1e-5
    dropout: float = 0.0

    # Rotary position embeddings
    rope_theta: float = 10000.0

    # Output head
    tie_word_embeddings: bool = False

    # Mixture-of-experts (optional). When enabled, the recurrent block's FFN is
    # replaced by a sparse MoE layer.
    use_moe: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_aux_loss_coef: float = 0.01

    # Truncated backprop through the recurrent iterations. Gradients flow
    # through at most this many iterations; the state is detached in between.
    # Must be >= 1 and <= n_recurrent.
    grad_recurrence_steps: int = 8

    # Recompute each recurrent iteration on the backward pass instead of
    # storing its activations. Trades compute for memory; the recurrent stack
    # reads this from the model config because it changes the forward pass.
    grad_checkpoint: bool = False

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must divide n_kv_heads"
        assert 1 <= self.grad_recurrence_steps <= self.n_recurrent, (
            "grad_recurrence_steps must lie in [1, n_recurrent]"
        )
        assert self.moe_top_k <= self.moe_num_experts

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ModelConfig":
        return cls(**json.loads(s))


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # Data
    train_files: list = field(default_factory=list)  # .txt files
    val_files: list = field(default_factory=list)
    seq_len: int = 1024

    # Optimisation
    batch_size: int = 8            # per-device micro-batch
    grad_accum_steps: int = 8     # effective batch = batch_size * grad_accum_steps
    max_steps: int = 20000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)

    # Precision / memory
    mixed_precision: bool = True  # bfloat16 autocast on CUDA; off on CPU

    # Logging / checkpointing
    log_every: int = 25
    eval_every: int = 500
    save_every: int = 2000
    out_dir: str = "checkpoints"
    seed: int = 1337

    def __post_init__(self) -> None:
        assert self.seq_len >= 16

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum_steps

    def to_json(self) -> str:
        d = asdict(self)
        d["betas"] = list(d["betas"])
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "TrainConfig":
        d = json.loads(s)
        d["betas"] = tuple(d["betas"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
def tiny_debug(model_overrides: dict | None = None,
               train_overrides: dict | None = None):
    """~1M params. Forward/backward/generate on CPU in seconds."""
    m = ModelConfig(
        vocab_size=512, max_seq_len=64, d_model=64, n_heads=4, n_kv_heads=2,
        d_ff=128, n_recurrent=2, tie_word_embeddings=True,
        grad_recurrence_steps=2,
    )
    t = TrainConfig(
        seq_len=32, batch_size=2, grad_accum_steps=1, max_steps=50,
        lr=1e-3, warmup_steps=5, mixed_precision=False,
        log_every=5, eval_every=25, save_every=50,
    )
    if model_overrides:
        for k, v in model_overrides.items():
            setattr(m, k, v)
        m.__post_init__()
    if train_overrides:
        for k, v in train_overrides.items():
            setattr(t, k, v)
    return m, t


def mini_60m(model_overrides: dict | None = None,
             train_overrides: dict | None = None):
    """Reference ~62M model sized for a single T4 (16 GB).

    Byte-level vocabulary keeps embeddings small, so capacity sits in the
    recurrent block. Parameter count is verified by ``benchmark.py``.
    """
    m = ModelConfig()  # defaults are the 62M configuration
    t = TrainConfig(
        seq_len=1024, batch_size=4, grad_accum_steps=16, max_steps=20000,
        lr=3e-4, warmup_steps=1000,
    )
    if model_overrides:
        for k, v in model_overrides.items():
            setattr(m, k, v)
        m.__post_init__()
    if train_overrides:
        for k, v in train_overrides.items():
            setattr(t, k, v)
    return m, t


PRESETS = {"tiny_debug": tiny_debug, "mini_60m": mini_60m}
