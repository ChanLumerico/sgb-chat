import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler

from tokenizers import Tokenizer

import pandas as pd
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        d_feedforward: int = 2048,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout_p: float = 0.1,
        pad_id: int = 0,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)

        self.positional_encoder = PositionalEncoding(d_model, dropout_p, max_len=5000)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_feedforward,
            dropout=dropout_p,
            batch_first=True,
        )

        self.out = nn.Linear(d_model, tgt_vocab_size, bias=not tie_weights)
        if tie_weights:
            self.out.weight = self.tgt_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _make_padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
        return tokens.eq(pad_id)

    def forward(
        self,
        src: torch.Tensor,
        tgt_inp: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        src_pad_mask: torch.Tensor = None,
        tgt_pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        device = src.device

        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt_inp) * math.sqrt(self.d_model)

        src_emb = self.positional_encoder(src_emb)
        tgt_emb = self.positional_encoder(tgt_emb)

        if tgt_mask is None:
            T = tgt_emb.size(1)
            tgt_mask = generate_square_subsequent_mask(T, device=device)

        if src_pad_mask is None:
            src_pad_mask = self._make_padding_mask(src, self.pad_id)
        if tgt_pad_mask is None:
            tgt_pad_mask = self._make_padding_mask(tgt_inp, self.pad_id)

        x = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
            tgt_mask=tgt_mask,
        )

        logits = self.out(x)
        return logits


model = Transformer(
    src_vocab_size=9000,
    tgt_vocab_size=9000,
    d_model=128,
    nhead=4,
    num_encoder_layers=4,
    num_decoder_layers=4,
    d_feedforward=512,
    dropout_p=0.3,
)


def loss_function(y_true, y_pred, pad_id=0):
    loss = F.cross_entropy(y_pred.transpose(1, 2), y_true, reduction="none")
    mask = (y_true != pad_id).float()
    loss = loss * mask
    return loss.sum() / mask.sum()


class TransformerScheduler(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self._step_count)
        scale = self.d_model**-0.5

        arg1 = step**-0.5
        arg2 = step * (self.warmup_steps**-1.5)

        lr = scale * min(arg1, arg2)
        return [lr for _ in self.base_lrs]


df = pd.read_csv("data/data.csv")
questions = [re.sub(r"([?.!,])", r" \1 ", str(s)).strip() for s in df["Q"]]
answers = [re.sub(r"([?.!,])", r" \1 ", str(s)).strip() for s in df["A"]]


tokenizer = Tokenizer.from_file("src/tokenizer.json")
PAD_ID = tokenizer.token_to_id("[PAD]")
START_ID = tokenizer.token_to_id("[START]")
END_ID = tokenizer.token_to_id("[END]")


def tokenize_and_filter(
    inputs,
    outputs,
    tokenizer,
    start_token_id,
    end_token_id,
    max_length,
    pad_id=0,
    device="cpu",
):
    tokenized_inputs, tokenized_outputs = [], []

    for s1, s2 in zip(inputs, outputs):
        # Encode and add start / end tokens
        s1_ids = [start_token_id] + tokenizer.encode(s1).ids + [end_token_id]
        s2_ids = [start_token_id] + tokenizer.encode(s2).ids + [end_token_id]

        # Truncate to max_length
        s1_ids = s1_ids[:max_length]
        s2_ids = s2_ids[:max_length]

        # Convert to torch tensors
        tokenized_inputs.append(torch.tensor(s1_ids, dtype=torch.long))
        tokenized_outputs.append(torch.tensor(s2_ids, dtype=torch.long))

    # Pad sequences
    tokenized_inputs = pad_sequence(
        tokenized_inputs, batch_first=True, padding_value=pad_id
    )
    tokenized_outputs = pad_sequence(
        tokenized_outputs, batch_first=True, padding_value=pad_id
    )

    # Ensure fixed length
    if tokenized_inputs.size(1) > max_length:
        tokenized_inputs = tokenized_inputs[:, :max_length]
    else:
        pad_len = max_length - tokenized_inputs.size(1)
        if pad_len > 0:
            tokenized_inputs = torch.nn.functional.pad(
                tokenized_inputs, (0, pad_len), value=pad_id
            )

    if tokenized_outputs.size(1) > max_length:
        tokenized_outputs = tokenized_outputs[:, :max_length]
    else:
        pad_len = max_length - tokenized_outputs.size(1)
        if pad_len > 0:
            tokenized_outputs = torch.nn.functional.pad(
                tokenized_outputs, (0, pad_len), value=pad_id
            )

    return tokenized_inputs.to(device), tokenized_outputs.to(device)


src, tgt = tokenize_and_filter(
    questions,
    answers,
    tokenizer=tokenizer,
    start_token_id=START_ID,
    end_token_id=END_ID,
    max_length=40,
    pad_id=PAD_ID,
)
