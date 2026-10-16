"""
OpenMythos 3B config — for the kyegomez/OpenMythos architecture.

Target: beat the best open ~3B models (Llama 3.2 3B, Qwen3, Phi-4-mini, Gemma, SmolLM2).
Reference variant: mythos_3b (dim=3072, MLA, 64 experts, 16 loop iters, 4k context).
See research/3b_landscape.md for the competitive analysis behind these choices.

Usage:
    from open_mythos.main import OpenMythos
    from configs.mythos_3b import get_config
    model = OpenMythos(get_config())
"""

from open_mythos.main import MythosConfig


def get_config() -> MythosConfig:
    return MythosConfig(
        vocab_size=32000,
        dim=3072,
        n_heads=24,
        n_kv_heads=6,
        max_seq_len=4096,
        max_loop_iters=16,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=384,
        q_lora_rank=768,
        qk_rope_head_dim=32,
        qk_nope_head_dim=96,
        v_head_dim=96,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=4096,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


# Training sketch (see training/ for the full script):
#   - Data: FineWeb-Edu class (~100B+ tokens target for 3B; see research/)
#   - Parallelism: FSDP + AdamW, bf16 mixed precision
#   - Schedule: linear warmup -> cosine decay
#   - Batch: ~4M tokens/step class (set by hardware; scale LR accordingly)
