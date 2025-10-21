import math
import re
import os

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim


class PositionalEncoding(nn.Module):
    def __init__(
        self, dim_model: int, dropout: float = 0.1, max_len: int = 5000
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = lucid.zeros(max_len, dim_model)
        position = lucid.arange(0, max_len, dtype=lucid.Float32).unsqueeze(axis=1)
        div_term = lucid.exp(
            lucid.arange(0, dim_model, 2, dtype=lucid.Float32)
            * (-lucid.log(10000.0) / dim_model)
        )

        pe[:, 0::2] = lucid.sin(position * div_term)
        pe[:, 0::2] = lucid.cos(position * div_term)

        pe = pe.unsqueeze(axis=0)
        self.register_buffer("pe", pe)


# TODO
NotImplemented
