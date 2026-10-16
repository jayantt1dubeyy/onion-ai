"""
OpenMythos 1B config — the "beat them at any cost" build.

Target class: TinyLlama-1.1B, SmolLM2-1.7B, OLMo-2-1B, Llama-3.2-1B, Qwen2.5-1.5B.
Verified: 1,056,094,338 params (forward pass smoke-tested on CPU).

Architecture decisions (see research/1b_battle_plan.md for the why):
  - GQA 16/4, NOT MLA (proven at this scale; MLA adds instability surface)
  - MoE: 64 experts / 8 active / 0 shared (Qwen3-style fine-grained, no shared expert)
  - 64K custom BPE (train on the actual data mix; embeddings already tied upstream)
  - 4K context, 16 max loop iters (train with loop-depth curriculum, see training/)
  - QK-Norm + sliding-window interleave: implemented via patches/ (not in upstream config)

Usage:
    from open_mythos.main import OpenMythos
    from configs.mythos_1b import get_config
    model = OpenMythos(get_config())
"""

from open_mythos.main import OpenMythos, MythosConfig


def get_config() -> MythosConfig:
    return MythosConfig(
        vocab_size=64000,          # custom BPE trained on our data mix
        dim=2048,
        n_heads=16,
        n_kv_heads=4,              # GQA 16/4
        max_seq_len=4096,
        max_loop_iters=16,         # inference max; train with curriculum (see below)
        prelude_layers=2,
        coda_layers=2,
        attn_type="gqa",
        # MoE: 64 fine-grained experts, 8 active per token, no shared expert
        n_experts=64,
        n_shared_experts=0,
        n_experts_per_tok=8,
        expert_dim=2048,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
        dropout=0.0,
    )


# Loop-depth curriculum (training-side, see training/train_1b.py):
#   steps 0–20%   : n_loops = 2   (cheap, stable; learn representations first)
#   steps 20–80%  : n_loops = 4
#   anneal phase  : n_loops = 8
#   eval          : sweep n_loops in {4, 8, 16} — test-time depth extrapolation
#                   is the killer feature no dense 1B model can match.
