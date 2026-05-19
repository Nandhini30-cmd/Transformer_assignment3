# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```
# DA6401 — Assignment 3: Attention Is All You Need

> Implementing the Transformer architecture from scratch for German → English Neural Machine Translation using the Multi30k dataset.

Links

Link to WandB Report: WandB Experiment Report (https://api.wandb.ai/links/mnandhini312-indian-institute-of-technology-madras/14uyhmvi)


Link to GitHub Repository: [Project GitHub Repository](https://github.com/Nandhini30-cmd/Transformer_assignment3)





## Overview

This project is a from-scratch PyTorch implementation of the Transformer model as described in [Vaswani et al., 2017 — "Attention Is All You Need"](https://arxiv.org/abs/1706.03762). The model is trained to translate German sentences to English on the [Multi30k](https://github.com/multi30k/dataset) dataset and evaluated using corpus-level BLEU score.

Key concepts implemented:
- Multi-Head Self-Attention & Cross-Attention
- Sinusoidal and Learned Positional Encodings
- Encoder-Decoder Transformer stack
- Noam Learning Rate Scheduler
- Label Smoothing Loss
- Greedy Decoding

---

## Project Structure

```
assignment3/
├── model.py        # Transformer architecture: Encoder, Decoder, Multi-Head Attention
├── dataset.py      # Multi30k dataset loading & spaCy tokenization
├── train.py        # Training loop, greedy decoding, BLEU evaluation, checkpointing
├── lr_scheduler.py # Noam warmup scheduler
├── requirements.txt
└── README.md
```

| File | Responsibility |
|------|---------------|
| `model.py` | `Transformer`, `Encoder`, `Decoder`, `MultiHeadAttention`, `PositionalEncoding`, masking utilities |
| `dataset.py` | `Multi30kDataset`, vocabulary building, spaCy tokenization, `collate_fn` |
| `train.py` | `run_epoch`, `greedy_decode`, `evaluate_bleu`, `save_checkpoint`, `load_checkpoint`, `LabelSmoothingLoss` |
| `lr_scheduler.py` | `NoamScheduler` — warmup + inverse-sqrt decay |

---


**Default Hyperparameters (base model):**

| Parameter | Value |
|-----------|-------|
| `d_model` | 512 |
| `num_heads` | 8 |
| `N` (layers) | 6 |
| `d_ff` | 2048 |
| `dropout` | 0.1 |
| `warmup_steps` | 4000 |
| `label_smoothing` | 0.1 |
| `max_len` | 5000 |

---

## Setup & Installation

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/da6401-assignment3.git
cd da6401-assignment3
```

**2. Create a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download spaCy language models**

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

---

## Dataset

The project uses the [Multi30k](https://github.com/multi30k/dataset) dataset — 31,014 German/English sentence pairs describing Flickr images.

| Split | Sentences |
|-------|-----------|
| Train | 29,000 |
| Validation | 1,014 |
| Test | 1,000 |

Tokenization is handled by **spaCy** (`de_core_news_sm` for German, `en_core_web_sm` for English). Special tokens added: `<pad>`, `<unk>`, `<sos>`, `<eos>`.

---

## Training

**Basic training run (defaults):**

```bash
python train.py \
  --epochs 15 \
  --batch_size 64 \
  --d_model 512 \
  --num_heads 8 \
  --N 6 \
  --d_ff 2048 \
  --dropout 0.1 \
  --warmup_steps 4000 \
  --save_path OP/best_model.pt \
  --save_path_last OP/last_model.pt \
  --experiment_name baseline
```

**With Wandb logging:**

```bash
python train.py \
  --epochs 15 \
  --wandb_api_key <YOUR_KEY> \
  --wandb_project DA6401_Assignment3 \
  --wandb_task_name baseline
```

### All CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 1.0 | Base learning rate (used with Noam) |
| `--warmup_steps` | 4000 | Noam scheduler warmup steps |
| `--d_model` | 512 | Model dimension |
| `--num_heads` | 8 | Number of attention heads |
| `--N` | 6 | Number of encoder/decoder layers |
| `--d_ff` | 2048 | Feed-forward hidden dimension |
| `--dropout` | 0.1 | Dropout probability |
| `--seed` | 42 | Random seed |
| `--patience` | 5 | Early stopping patience |
| `--device` | cuda | Device (`cuda` / `cpu`) |
| `--scheduler_type` | noam | `noam` or `fixed` LR |
| `--pos_encoding` | sinusoidal | `sinusoidal` or `learned` |
| `--label_smoothing` | 0.1 | Label smoothing factor ε |
| `--use_scaling` | 1 | Attention score scaling (1=on, 0=off) |
| `--save_attention_maps` | 0 | Log attention heatmaps to Wandb |
| `--experiment_name` | baseline | Name prefix for CSV log file |


```

Logs per-head encoder self-attention heatmaps to Wandb each epoch.

---



---

Each checkpoint stores:

```python
{
    "epoch":                int,
    "model_state_dict":     OrderedDict,
    "optimizer_state_dict": OrderedDict,
    "scheduler_state_dict": OrderedDict,
    "model_config":         dict,   # all kwargs to reconstruct Transformer
    "src_vocab":            Vocab,
    "tgt_vocab":            Vocab,
}
```



---

## Wandb Logging

When `--wandb_api_key` is provided, the following metrics are logged per epoch:

| Metric | Description |
|--------|-------------|
| `train_loss` | Average training loss |
| `val_loss` | Average validation loss |
| `val_bleu` | Validation BLEU score |
| `learning_rate` | Current LR from scheduler |
| `prediction_confidence` | Mean softmax probability on correct tokens |
| `query_grad_norm` | Query weight gradient norm (first 1000 steps) |
| `key_grad_norm` | Key weight gradient norm (first 1000 steps) |
| `attention_head_*` | Encoder self-attention heatmaps (if enabled) |
| `test_bleu` | Final test BLEU (end of run) |

---

## Results

| Experiment | Best Val BLEU | Notes |
|------------|--------------|-------|
| Baseline (sinusoidal) | — | `d_model=512, N=6, heads=8` |
| Learned PE | — | `nn.Embedding` positional encoding |
| No label smoothing | — | `ε=0.0` |
| Fixed LR | — | `Adam lr=1e-4`, no warmup |
| No attention scaling | — | Removed `√d_k` divisor |



---

