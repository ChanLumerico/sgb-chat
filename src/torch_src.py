import math
import re
import os, glob

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader, random_split

from tokenizers import Tokenizer

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm


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
        n_heads: int = 8,
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
            nhead=n_heads,
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
        return tokens == pad_id

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
            tgt_mask = tgt_mask.bool()

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

tokenizer = Tokenizer.from_file("data/tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
max_length = 40

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
        s1_ids = [start_token_id] + tokenizer.encode(s1).ids + [end_token_id]
        s2_ids = [start_token_id] + tokenizer.encode(s2).ids + [end_token_id]

        s1_ids = s1_ids[:max_length]
        s2_ids = s2_ids[:max_length]

        tokenized_inputs.append(torch.tensor(s1_ids, dtype=torch.long))
        tokenized_outputs.append(torch.tensor(s2_ids, dtype=torch.long))

    tokenized_inputs = pad_sequence(
        tokenized_inputs, batch_first=True, padding_value=pad_id
    )
    tokenized_outputs = pad_sequence(
        tokenized_outputs, batch_first=True, padding_value=pad_id
    )

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
    max_length=max_length,
    pad_id=PAD_ID,
)


device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 64
num_layers = 2
d_model = 256
n_heads = 8
d_feedforward = 512
dropout_p = 0.1

dec_inputs = tgt[:, :-1]
dec_labels = tgt[:, 1:]

dataset = TensorDataset(src, dec_inputs, dec_labels)

val_ratio = 0.1
n_total = len(dataset)
n_val = int(n_total * val_ratio)
n_train = n_total - n_val

train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
    pin_memory=(device == "cuda"),
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=2,
    pin_memory=(device == "cuda"),
)

model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    num_encoder_layers=num_layers,
    num_decoder_layers=num_layers,
    d_feedforward=d_feedforward,
    dropout_p=dropout_p,
)


def seq_ce_loss(logits, targets, pad_id=0):

    loss = F.cross_entropy(
        logits.transpose(1, 2),
        targets,
        reduction="none",
        ignore_index=pad_id,
    )

    valid = (targets != pad_id).float()
    return (loss * valid).sum() / (valid.sum() + 1e-8)


@torch.no_grad()
def token_accuracy(logits, targets, pad_id=0):

    preds = logits.argmax(dim=-1)
    mask = targets != pad_id
    correct = (preds.eq(targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / max(1, total)


@torch.no_grad()
def evaluate_loss(model, dataloader, pad_id=0):
    model.eval()
    total, count = 0.0, 0
    for src, dec_inp, dec_out in dataloader:
        src, dec_inp, dec_out = src.to(device), dec_inp.to(device), dec_out.to(device)

        logits = model(
            src=src,
            tgt_inp=dec_inp,
            src_pad_mask=(src == pad_id),
            tgt_pad_mask=(dec_inp == pad_id),
        )
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            dec_out,
            reduction="none",
            ignore_index=pad_id,
        )
        mask = (dec_out != pad_id).float()
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        total += loss.item()
        count += 1
    return total / max(1, count)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    epoch_loss,
    batch_losses,
    epoch_losses,
    val_epoch_losses=None,
    path="checkpoints",
):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "epoch_loss": epoch_loss,
        "batch_losses": batch_losses,
        "epoch_losses": epoch_losses,
        "val_epoch_losses": val_epoch_losses,
    }
    torch.save(checkpoint, os.path.join(path, f"epoch_{epoch}.pt"))


def load_latest_checkpoint(ckpt_dir, model, optimizer, scheduler, device):
    if not os.path.exists(ckpt_dir):
        return 1, [], [], []

    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not ckpts:
        return 1, [], [], []

    latest_ckpt = sorted(ckpts, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
    checkpoint = torch.load(os.path.join(ckpt_dir, latest_ckpt), map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    return (
        checkpoint["epoch"] + 1,
        checkpoint.get("batch_losses", []),
        checkpoint.get("epoch_losses", []),
        checkpoint.get("val_epoch_losses", []),
    )


def train(
    model,
    dataloader,
    optimizer,
    scheduler=None,
    pad_id=0,
    device="cuda",
    epochs=50,
    grad_clip=None,
    ckpt_dir="checkpoints",
    start_epoch=1,
    batch_losses=None,
    epoch_losses=None,
    val_loader=None,
    early_stop_patience=None,
):
    model.to(device)
    batch_losses = [] if batch_losses is None else list(batch_losses)
    epoch_losses = [] if epoch_losses is None else list(epoch_losses)
    val_epoch_losses = []

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss, total_acc = 0.0, 0.0
        n_batches = len(dataloader)

        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{epochs}",
            leave=True,
            ncols=100,
            dynamic_ncols=True,
        )
        for step, (src, dec_inp, dec_out) in enumerate(progress, start=1):
            src, dec_inp, dec_out = (
                src.to(device),
                dec_inp.to(device),
                dec_out.to(device),
            )

            logits = model(
                src=src,
                tgt_inp=dec_inp,
                src_pad_mask=(src == pad_id),
                tgt_pad_mask=(dec_inp == pad_id),
            )

            loss = seq_ce_loss(logits, dec_out, pad_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            if scheduler:
                scheduler.step()

            acc = token_accuracy(logits, dec_out, pad_id)
            total_loss += loss.item()
            total_acc += acc
            batch_losses.append(loss.item())

            progress.set_postfix(
                {
                    "loss": f"{total_loss/step:.4f}",
                    "acc": f"{total_acc/step:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        train_epoch_loss = total_loss / n_batches
        epoch_losses.append(train_epoch_loss)

        if val_loader is not None:
            val_loss = evaluate_loss(model, val_loader, pad_id=pad_id)
            val_epoch_losses.append(val_loss)
            print(
                f"[Epoch {epoch}] Valid Loss: {val_loss:.4f} | "
                f"Perplexity: {math.exp(val_loss):.2f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    train_epoch_loss,
                    batch_losses,
                    epoch_losses,
                    path=os.path.join(ckpt_dir, "best"),
                )
                bad_epochs = 0

            else:
                bad_epochs += 1
                if (early_stop_patience is not None) and (
                    bad_epochs >= early_stop_patience
                ):
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(no val improvement for {bad_epochs} epochs)."
                    )
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        train_epoch_loss,
                        batch_losses,
                        epoch_losses,
                        path=ckpt_dir,
                    )
                    return batch_losses, epoch_losses, val_epoch_losses

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            train_epoch_loss,
            batch_losses,
            epoch_losses,
            val_epoch_losses=val_epoch_losses,
            path=ckpt_dir,
        )

    return batch_losses, epoch_losses, val_epoch_losses


optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
scheduler = TransformerScheduler(optimizer, d_model=d_model, warmup_steps=4000)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
    pin_memory=(device == "cuda"),
)

start_epoch, batch_losses, epoch_losses, val_epoch_losses = load_latest_checkpoint(
    ckpt_dir="checkpoints",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
)

batch_losses, epoch_losses, val_epoch_losses = train(
    model,
    dataloader=train_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    pad_id=PAD_ID,
    device=device,
    epochs=50,
    grad_clip=1.0,
    ckpt_dir="checkpoints",
    start_epoch=start_epoch,
    batch_losses=batch_losses,
    epoch_losses=epoch_losses,
    val_loader=val_loader,
    early_stop_patience=5,
)


@torch.no_grad()
def evaluate(sentence):
    model.eval()
    sentence_ids = tokenizer.encode(sentence).ids
    src = torch.tensor(
        [[START_ID] + sentence_ids + [END_ID]], dtype=torch.long, device=device
    )

    output = torch.tensor([[START_ID]], dtype=torch.long, device=device)
    for _ in range(max_length):
        src_pad_mask = src == tokenizer.token_to_id("[PAD]")
        tgt_pad_mask = output == tokenizer.token_to_id("[PAD]")
        tgt_mask = model.transformer.generate_square_subsequent_mask(output.size(1)).to(
            device
        )

        logits = model(
            src=src,
            tgt_inp=output,
            src_pad_mask=src_pad_mask,
            tgt_pad_mask=tgt_pad_mask,
            tgt_mask=tgt_mask,
        )

        next_token_logits = logits[:, -1, :]
        predicted_id = next_token_logits.argmax(dim=-1).item()

        if predicted_id == END_ID:
            break

        next_token = torch.tensor([[predicted_id]], device=device)
        output = torch.cat([output, next_token], dim=-1)

    return output.squeeze(0).tolist()


@torch.no_grad()
def predict(sentence):
    predicted_ids = evaluate(sentence)
    decoded = tokenizer.decode(
        [i for i in predicted_ids if i < tokenizer.get_vocab_size()]
    )
    return decoded


print("Model Prepared.\n")

while True:
    input_ = input("In: ").strip()
    if input_ == "END":
        break

    output = predict(input_)
    print(f"Out: {output}\n")
