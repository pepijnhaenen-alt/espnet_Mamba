"""Streaming S4 encoder for ESPnet2 ASR."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.state_spaces.model import SequenceModel


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
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class S4Encoder(AbsEncoder):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        hidden_size: int = 512,
        num_blocks: int = 12,
        d_state: int = 64,
        l_max: Optional[int] = None,
        channels: int = 1,
        bidirectional: bool = False,
        activation: str = "gelu",
        postact: str = "glu",
        hyper_act: Optional[str] = None,
        bottleneck: Optional[int] = None,
        gate: Optional[int] = None,
        conv_kernel_size: int = 15,
        conv_num_layers: int = 2,
        dropout_rate: float = 0.1,
        chunk_size: int = 0,
        s4_conf: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._output_size = output_size
        self.chunk_size = chunk_size

        self.in_proj = nn.Linear(input_size, hidden_size)
        self.frontend_convs = nn.ModuleList(
            [
                _CausalDepthwiseConv1d(hidden_size, conv_kernel_size, dropout_rate)
                for _ in range(conv_num_layers)
            ]
        )

        layer_conf: Dict[str, Any] = {
            "_name_": "s4",
            "d_state": d_state,
            "l_max": l_max,
            "channels": channels,
            "bidirectional": bidirectional,
            "activation": activation,
            "postact": postact,
            "hyper_act": hyper_act,
            "bottleneck": bottleneck,
            "gate": gate,
            "dropout": dropout_rate,
            "transposed": False,
        }
        if s4_conf is not None:
            layer_conf.update(s4_conf)

        self.blocks = SequenceModel(
            hidden_size,
            n_layers=num_blocks,
            transposed=False,
            dropout=dropout_rate,
            prenorm=True,
            layer=layer_conf,
            residual={"_name_": "residual"},
            norm="layer",
            dropinp=0.0,
            drop_path=0.0,
        )
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

        x = x.contiguous()
        x, new_states = self.blocks(x, state=prev_states)
        x = self.out_proj(x)

        return x, ilens, new_states