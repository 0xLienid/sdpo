"""
Non-causal (bidirectional) correction head for SDPO distillation targets.

Takes the base model's causal hidden states (each position only sees preceding
context) and applies bidirectional self-attention so each position can attend
to all others. Outputs per-position correction vectors that, when added to the
original hidden states and passed through lm_head, produce improved logits.

The output projection is initialized near-zero so the model starts close to
the original teacher's behavior.
"""

import torch
import torch.nn as nn


class CorrectionHead(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 16,
        ff_mult: int = 4,
        dropout: float = 0.0,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.normal_(self.output_proj.weight, std=init_scale)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim) — post-norm base model states
        Returns:
            corrections: (batch, seq_len, hidden_dim) — additive correction vectors
        """
        x = self.input_norm(hidden_states)
        x = self.encoder(x)
        return self.output_proj(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
