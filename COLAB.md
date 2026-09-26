# Running the full experiment on a Colab T4

The reference model (`mini_60m`, ~62.9M params) needs a GPU. This runbook
takes you from zero to a trained checkpoint on a single T4 (16 GB).

## 1. Setup (one cell)

```python
!git clone <your-openmythos-repo-url> && cd openmythos
!pip install -q -r requirements.txt
```

Check the GPU: `!nvidia-smi` should show a Tesla T4.

## 2. Sanity check (one cell, ~2 min)

```python
!cd openmythos && python -m openmythos.benchmark --preset tiny_debug --mode train --steps 5
!cd openmythos && python -m openmythos.benchmark --preset mini_60m --mode vram
```

The first command runs forward/backward/optimizer steps on CPU-cheap settings;
the second prints the analytic VRAM estimate (~2.4 GB — comfortable on a T4).
If either fails, do not start the long run.

## 3. Data (one cell)

TinyStories is the recommended corpus for this scale (simple language,
designed for small models):

```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="train")
with open("data/train.txt", "w") as f:
    for ex in ds:
        f.write(ex["text"].strip() + "\n")
ds = load_dataset("roneneldan/TinyStories", split="validation")
with open("data/val.txt", "w") as f:
    for ex in ds:
        f.write(ex["text"].strip() + "\n")
```

## 4. Train (one cell, starts the run)

```python
!cd openmythos && nohup python -m openmythos.train --preset mini_60m \
    --train-data data/train.txt --val-data data/val.txt \
    --out-dir checkpoints > train.log 2>&1 &
```

Default config: seq 1024, micro-batch 4, 16x accumulation (effective batch
256 sequences), 20k steps, warmup 1k, cosine decay. Watch `train.log`;
loss is also appended as JSONL to `checkpoints/loss.jsonl`.

Expected: loss falls from ~ln(260) ≈ 5.6 toward ~2.5-3.5 on TinyStories.
If loss stalls above 4 after 2k steps, stop and check the data pipeline.

## 5. Resume after a preemption

```python
!cd openmythos && nohup python -m openmythos.train --preset mini_60m \
    --train-data data/train.txt --val-data data/val.txt \
    --resume checkpoints/latest.pt > train.log 2>&1 &
```

Checkpoints store model, optimizer, scaler, and RNG state, so resuming is
bit-for-bit continuous.

## 6. Evaluate and generate

```python
!cd openmythos && python -m openmythos.evaluate \
    --ckpt checkpoints/final.pt --text data/val.txt
```

```python
from openmythos.model import OpenMythosModel
from openmythos.config import ModelConfig
from openmythos.tokenizer import ByteTokenizer
from openmythos.inference import generate
import torch
ckpt = torch.load("checkpoints/final.pt", map_location="cuda", weights_only=False)
model = OpenMythosModel(ModelConfig.from_json(ckpt["model_cfg"])).cuda().eval()
model.load_state_dict(ckpt["model"])
print(generate(model, ByteTokenizer(), "Once upon a time",
               max_new_tokens=100, device="cuda"))
```

## 7. The recurrence experiment (the paper's core question)

Once the baseline trains, the interesting comparison is recurrence depth vs.
parameters. Train variants with the same config but different `n_recurrent`
by editing the preset call in `train.py`, e.g.:

| Variant | n_recurrent | Params | Question |
|---|---|---|---|
| shallow | 2 | ~62.9M | baseline depth |
| reference | 8 | ~62.9M | does iteration help? |
| deep | 16 | ~62.9M | diminishing returns? |

Parameter count is identical across rows (weights are shared) — any
perplexity difference is pure compute-depth effect. That is the experiment
that turns this codebase into a paper.
