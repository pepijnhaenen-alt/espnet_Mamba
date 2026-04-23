"""Streaming Mamba encoder for ESPnet2 ASR.

This encoder is intentionally thin and delegates the state-space implementation
to ``espnet2.asr.state_spaces.s6`` which wraps the upstream ``mamba_ssm``
package for both Mamba1 and Mamba2.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.state_spaces.s6 import Mamba1, Mamba2


class _CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, groups=channels, bias=True
        )
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = x.transpose(1, 2)               # (B, C, T)
        x = F.pad(x, (self.kernel_size - 1, 0))  # strictly causal (left pad only)
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class MambaEncoder(AbsEncoder):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        hidden_size: int = 512,
        num_blocks: int = 12,
        mamba_type: str = "mamba1",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mamba2_d_ssm: Optional[int] = None,
        mamba2_headdim: int = 64,
        mamba2_ngroups: int = 1,
        mamba2_chunk_size: int = 256,
        conv_kernel_size: int = 15,
        conv_num_layers: int = 2,
        linear_units: int = 2048,
        dropout_rate: float = 0.1,
        chunk_size: int = 0,
        block_conf: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._output_size = output_size
        self.chunk_size = chunk_size
        self.mamba_type = mamba_type.lower()

        self.in_proj = nn.Linear(input_size, hidden_size)

        self.frontend_convs = nn.ModuleList(
            [
                _CausalDepthwiseConv1d(hidden_size, conv_kernel_size, dropout_rate)
                for _ in range(conv_num_layers)
            ]
        )

        if self.mamba_type not in {"mamba1", "mamba2"}:
            raise ValueError(f"Unsupported mamba_type={mamba_type}. Use mamba1 or mamba2.")

        block_conf = {} if block_conf is None else dict(block_conf)
        if self.mamba_type == "mamba1":
            # Keep a CPU-compatible default; users can override via block_conf.
            block_conf.setdefault("use_fast_path", False)
        else:
            # Keep a CPU-compatible default; users can override via block_conf.
            block_conf.setdefault("use_mem_eff_path", False)
        block_cls = Mamba1 if self.mamba_type == "mamba1" else Mamba2

        self.blocks = nn.ModuleList()
        self.mamba_norms = nn.ModuleList()
        for layer_idx in range(num_blocks):
            if block_cls is Mamba1:
                block = block_cls(
                    d_model=hidden_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    layer_idx=layer_idx,
                    **block_conf,
                )
            else:
                block = block_cls(
                    d_model=hidden_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    d_ssm=mamba2_d_ssm,
                    headdim=mamba2_headdim,
                    ngroups=mamba2_ngroups,
                    chunk_size=mamba2_chunk_size,
                    layer_idx=layer_idx,
                    **block_conf,
                )
            self.blocks.append(block)
            self.mamba_norms.append(nn.LayerNorm(hidden_size))
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_blocks)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, linear_units),
                    nn.SiLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(linear_units, hidden_size),
                    nn.Dropout(dropout_rate),
                )
                for _ in range(num_blocks)
            ]
        )

        self.norm_out = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: Optional[List[Any]] = None,
        ctc=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Any]]]:
        x = self.in_proj(xs_pad)

        for conv in self.frontend_convs:
            x = conv(x)

        new_states: List[Any] = []
        for i, (blk, ffn) in enumerate(zip(self.blocks, self.ffns)):
            st = None if prev_states is None else prev_states[i]

            # Pre-norm + residual around the Mamba mixer for stable training.
            x_mamba = self.mamba_norms[i](x).contiguous()
            x_mamba, st = blk(x_mamba, state=st)
            x = x + x_mamba

            # Pre-norm + residual around the FFN, transformer-style.
            x = x + ffn(self.ffn_norms[i](x))
            new_states.append(st)

        x = self.norm_out(x)
        x = self.out_proj(x)

        return x, ilens, new_states