#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pickle
import numpy as np
from pathlib import Path
import torch
import sys

sys.path.append(str((Path.cwd() / "nanochat").resolve()))
from nanochat.gpt import GPT, GPTConfig


# In[13]:


data_dir = Path('data/processed/')
meta = pickle.loads((data_dir / 'meta.pkl').read_bytes())
vocab_size = meta['vocab_size']
context_length = meta['context_length']
train_tokens = np.fromfile(data_dir / 'train.bin', dtype=np.uint16)
val_tokens = np.fromfile(data_dir / 'val.bin', dtype=np.uint16)
(vocab_size, context_length, len(train_tokens), len(val_tokens))

# In[14]:


def nanogpt_iter(data, block_size, batch_size):
    max_start = data.size(0) - block_size - 1
    idx = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    return x, y


# In[15]:


device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
config = GPTConfig(sequence_len=context_length, vocab_size=vocab_size, n_layer=8, n_head=8, n_kv_head=4, n_embd=512)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

start_step = 1
for ckpt_path in (
    Path("/data/chess_min.pt"), # stored in modal volume
    # Path("chess_min.pt"), # stored in local directory
):
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        saved_config = state.get('meta', {}).get('model_config', {})
        if saved_config.get('n_layer') == config.n_layer and saved_config.get('n_embd') == config.n_embd:
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            start_step = state['step'] + 1
            print(f"Loaded checkpoint from {ckpt_path}")
            break
        else:
            print("Config mismatch, starting fresh")
    else:
        print("No checkpoint found, starting fresh")

# In[16]:


train_data = torch.from_numpy(train_tokens.astype(np.int64))
val_data = torch.from_numpy(val_tokens.astype(np.int64))
max_iters = 50000
eval_interval = 100
batch_size = 32
for step in range(start_step, max_iters + 1):
    xb_cpu, yb_cpu = nanogpt_iter(train_data, context_length, batch_size)
    xb = xb_cpu.to(device, non_blocking=True)
    yb = yb_cpu.to(device, non_blocking=True)
    logits = model(xb)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if step % eval_interval == 0 or step == 1:
        xb_val_cpu, yb_val_cpu = nanogpt_iter(val_data, context_length, batch_size)
        xb_val = xb_val_cpu.to(device, non_blocking=True)
        yb_val = yb_val_cpu.to(device, non_blocking=True)
        with torch.no_grad():
            logits_val = model(xb_val)
            val_loss = torch.nn.functional.cross_entropy(logits_val.view(-1, logits_val.size(-1)), yb_val.view(-1))
        print(f'step {step:04d}/{max_iters} train {loss.item():.4f} val {val_loss.item():.4f}')


    if step % 1000 == 0:
        checkpoint = {
            'model': model.state_dict(),
            'meta': {'model_config': config.__dict__, 'tokenizer': meta},
            'optimizer': optimizer.state_dict(),
            'step': step,
        }
        torch.save(checkpoint, '/data/chess_min.pt')
        print(f'Saved checkpoint to chess_min.pt at step {step}')