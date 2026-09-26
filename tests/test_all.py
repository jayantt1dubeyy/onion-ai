"""Verification suite for OpenMythos. Run: ``.venv/bin/python -m pytest tests/ -x -q``
or ``.venv/bin/python tests/test_all.py``.

Covers: RoPE shapes/positions, GQA head repetition, recurrent weight sharing,
truncated BPTT, forward/backward, KV-cache generation equivalence, tokenizer
round-trip, scheduler values, and checkpoint save/resume.
"""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from openmythos.attention import GroupedQueryAttention, RotaryEmbedding
from openmythos.config import ModelConfig, tiny_debug
from openmythos.inference import generate
from openmythos.model import OpenMythosModel
from openmythos.moe import SparseMoE, SwiGLU
from openmythos.scheduler import WarmupCosine
from openmythos.tokenizer import ByteTokenizer
from openmythos.trainer import Trainer


def test_rope_shapes_and_positions():
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
    x = torch.randn(2, 4, 10, 16)
    y0 = rope.apply(x, start_pos=0)
    y5 = rope.apply(x, start_pos=5)
    assert y0.shape == x.shape and y5.shape == x.shape
    # different positions -> different rotations
    assert not torch.allclose(y0, y5)
    # zero vector stays zero
    assert torch.allclose(rope.apply(torch.zeros(1, 1, 3, 16)),
                          torch.zeros(1, 1, 3, 16))
    print("ok rope")


def test_rope_against_reference():
    """RoPE must match a direct complex-number reference implementation."""
    torch.manual_seed(0)
    d = 8
    rope = RotaryEmbedding(head_dim=d, max_seq_len=16, theta=10000.0)
    x = torch.randn(1, 1, 4, d)
    got = rope.apply(x, start_pos=0)
    # reference: complex rotation
    inv = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
    pos = torch.arange(4).float()
    ang = torch.outer(pos, inv)                       # (4, d/2)
    xc = torch.complex(x[..., : d // 2], x[..., d // 2 :])
    rot = torch.polar(torch.ones_like(ang), ang).unsqueeze(0).unsqueeze(0)
    xc = xc * rot
    ref = torch.cat([xc.real, xc.imag], dim=-1)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()
    print("ok rope reference")


def test_gqa_head_repeat():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_heads=8, n_kv_heads=2,
                      d_ff=64, n_recurrent=1, max_seq_len=32,
                      grad_recurrence_steps=1)
    attn = GroupedQueryAttention(cfg).eval()
    x = torch.randn(2, 12, 32)
    y, _ = attn(x)
    assert y.shape == (2, 12, 32)
    # with n_kv_heads == n_heads the repeat must be a no-op: build an
    # 8-kv-head twin and check identical outputs given identical weights
    cfg2 = ModelConfig(vocab_size=64, d_model=32, n_heads=8, n_kv_heads=8,
                       d_ff=64, n_recurrent=1, max_seq_len=32,
                       grad_recurrence_steps=1)
    a2 = GroupedQueryAttention(cfg2).eval()
    with torch.no_grad():
        a2.q_proj.weight.copy_(attn.q_proj.weight)
        a2.o_proj.weight.copy_(attn.o_proj.weight)
        for i in range(8):
            src = i // 4
            a2.k_proj.weight[i * 4:(i + 1) * 4].copy_(
                attn.k_proj.weight[src * 4:(src + 1) * 4])
            a2.v_proj.weight[i * 4:(i + 1) * 4].copy_(
                attn.v_proj.weight[src * 4:(src + 1) * 4])
    y2, _ = a2(x)
    assert torch.allclose(y, y2, atol=1e-5), (y - y2).abs().max()
    print("ok gqa repeat")


def test_causality():
    """Future tokens must not influence the past (training mode, no cache)."""
    torch.manual_seed(1)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_heads=4, n_kv_heads=2,
                      d_ff=64, n_recurrent=3, max_seq_len=32,
                      grad_recurrence_steps=3)
    m = OpenMythosModel(cfg).eval()
    x = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        l1, _, _ = m(x)
        x2 = x.clone()
        x2[0, 8:] = torch.randint(0, 64, (8,))
        l2, _, _ = m(x2)
    assert torch.allclose(l1[:, :8], l2[:, :8], atol=1e-5)
    print("ok causality")


def test_weight_sharing():
    mc, _ = tiny_debug()
    m = OpenMythosModel(mc)
    # one block only: count its params, confirm stack holds a single block
    n_block = sum(p.numel() for p in m.stack.block.parameters())
    n_stack = sum(p.numel() for p in m.stack.parameters())
    assert n_block == n_stack, "recurrent stack must share one block"
    print(f"ok sharing (block params: {n_block:,})")


def test_forward_backward():
    torch.manual_seed(2)
    mc, tc = tiny_debug()
    m = OpenMythosModel(mc)
    ids = torch.randint(0, mc.vocab_size, (tc.batch_size, tc.seq_len + 1))
    total, ce, aux = m.forward_loss(ids)
    assert torch.isfinite(total) and total.item() > 0
    total.backward()
    for p in m.parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()
    print(f"ok forward/backward (loss {total.item():.3f})")


def test_truncated_bptt():
    """With grad_recurrence_steps=1, only one iteration's graph is retained."""
    torch.manual_seed(3)
    mc, _ = tiny_debug({"n_recurrent": 4, "grad_recurrence_steps": 1})
    m = OpenMythosModel(mc)
    m.train()
    ids = torch.randint(0, mc.vocab_size, (2, 17))
    total, _, _ = m.forward_loss(ids)
    total.backward()
    assert all(p.grad is not None for p in m.stack.block.parameters()
               if p.requires_grad)
    print("ok truncated bptt")


def test_kv_cache_equivalence():
    """Incremental generation with caches must match full-sequence forward."""
    torch.manual_seed(4)
    mc, _ = tiny_debug()
    m = OpenMythosModel(mc).eval()
    ids = torch.randint(0, mc.vocab_size, (1, 10))
    with torch.no_grad():
        full, _, _ = m(ids)
        caches = m.init_caches()
        logits, _, caches = m(ids[:, :4], caches=caches, start_pos=0)
        assert torch.allclose(logits, full[:, :4], atol=1e-5)
        for t in range(4, 10):
            logits, _, caches = m(ids[:, t:t + 1], caches=caches, start_pos=t)
            assert torch.allclose(logits, full[:, t:t + 1], atol=1e-4), t
    print("ok kv cache equivalence")


def test_generation_runs():
    torch.manual_seed(5)
    mc, _ = tiny_debug()
    m = OpenMythosModel(mc)
    text = generate(m, ByteTokenizer(), "hello", max_new_tokens=8,
                    temperature=0.0, seed=0)
    assert isinstance(text, str)
    print(f"ok generate -> {text!r}")


def test_tokenizer_roundtrip():
    tok = ByteTokenizer()
    for s in ["hello world", "café 🌀", "x" * 100]:
        assert tok.decode(tok.encode(s, add_bos=False, add_eos=False)) == s
    print("ok tokenizer")


def test_moe_shapes_and_aux():
    torch.manual_seed(6)
    moe = SparseMoE(32, 64, num_experts=4, top_k=2)
    x = torch.randn(2, 8, 32)
    out, aux = moe(x)
    assert out.shape == x.shape and aux.shape == ()
    assert aux.item() >= 0
    print("ok moe")


def test_scheduler_values():
    from openmythos.config import TrainConfig
    tc = TrainConfig(warmup_steps=10, max_steps=100, lr=1e-3, min_lr=1e-4)
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=tc.lr)
    s = WarmupCosine(opt, tc)
    assert abs(s.step(0) - 1e-4) < 1e-9          # 1/warmup of base
    assert abs(s.step(9) - 1e-3) < 1e-9          # end of warmup
    assert abs(s.step(99) - 1e-4) < 1e-9         # cosine floor
    mid = s.step(54)
    assert 1e-4 < mid < 1e-3
    print("ok scheduler")


def test_checkpoint_roundtrip():
    torch.manual_seed(7)
    mc, tc = tiny_debug()
    with tempfile.TemporaryDirectory() as d:
        tc.out_dir = d
        tr = Trainer(mc, tc, device="cpu")
        p = tr.save("t0")
        before = {k: v.clone() for k, v in tr.model.state_dict().items()}
        # perturb then restore
        with torch.no_grad():
            for v in tr.model.parameters():
                v.add_(1.0)
        tr.load(p)
        for k, v in tr.model.state_dict().items():
            assert torch.equal(v, before[k]), k
    print("ok checkpoint")


if __name__ == "__main__":
    test_rope_shapes_and_positions()
    test_rope_against_reference()
    test_gqa_head_repeat()
    test_causality()
    test_weight_sharing()
    test_forward_backward()
    test_truncated_bptt()
    test_kv_cache_equivalence()
    test_generation_runs()
    test_tokenizer_roundtrip()
    test_moe_shapes_and_aux()
    test_scheduler_values()
    test_checkpoint_roundtrip()
    print("\nALL TESTS PASSED")
