from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial
import torch
from torch.utils.data import DataLoader

def prepare_dataloaders(max_len=128, batch_size=64, limit_train_samples=20000):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '</s>'})

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    dataset = load_dataset("iwslt2017", "iwslt2017-en-de",
                           split={'train': 'train', 'validation': 'validation'},
                           trust_remote_code=True)

    train_raw, val_raw = dataset['train'], dataset['validation']
    if limit_train_samples > 0:
        train_raw = train_raw.select(range(min(limit_train_samples, len(train_raw))))

    def tokenize_batch(examples):
        en = [t['en'] for t in examples['translation']]
        de = [t['de'] for t in examples['translation']]
        enc = tokenizer(en, truncation=True, padding=False, max_length=max_len)
        dec = tokenizer(de, truncation=True, padding=False, max_length=max_len - 1)
        return {"input_ids": enc["input_ids"], "labels": [ids + [eos_id] for ids in dec["input_ids"]]}

    train_tok = train_raw.map(partial(tokenize_batch), batched=True, remove_columns=train_raw.column_names)
    val_tok = val_raw.map(partial(tokenize_batch), batched=True, remove_columns=val_raw.column_names)

    def collate_fn(batch):
        input_ids = [b['input_ids'] for b in batch]
        labels = [b['labels'] for b in batch]
        max_src = max(len(x) for x in input_ids)
        max_tgt = max(len(x) for x in labels)
        src_padded = [x + [pad_id] * (max_src - len(x)) for x in input_ids]
        tgt_padded = [x + [pad_id] * (max_tgt - len(x)) for x in labels]
        return {
            "src_input": torch.tensor(src_padded, dtype=torch.long),
            "tgt_input": torch.tensor(tgt_padded, dtype=torch.long)
        }

    train_loader = DataLoader(train_tok, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_tok, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, tokenizer
