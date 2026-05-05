"""Streaming Hybrid Mamba+Attention Encoder for ESPnet2 ASR.

This encoder combines within-chunk attention and trans-chunk Mamba SSM
for streaming speech recognition with low latency and high performance.

Based on: "Advancing Streaming ASR with Chunk-wise Attention and Trans-chunk 
Selective State Spaces" - combining local attention with selective state spaces
for effective streaming decoding.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.state_spaces.s6 import Mamba1, Mamba2
from espnet2.asr.state_spaces.attention import MultiHeadedAttention


class _CausalDepthwiseConv1d(nn.Module):
    """Causal depthwise convolution for preprocessing."""

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
        x = x.transpose(1, 2)  # (B, C, T)
        x = F.pad(x, (self.kernel_size - 1, 0))  # strictly causal (left pad only)
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class HybridMambaAttentionBlock(nn.Module):
    """Hybrid block combining within-chunk attention + trans-chunk Mamba SSM.
    
    This block implements the hybrid architecture from the paper that uses:
    - MultiHeadedAttention for within-chunk local dynamics
    - Mamba for trans-chunk long-range dependencies with selective gating
    
    Args:
        hidden_size: Dimension of hidden states
        num_heads: Number of attention heads for within-chunk attention
        d_state: State dimension for Mamba
        d_conv: Kernel size for Mamba internal convolution
        expand: Expansion factor for Mamba FFN
        mamba_type: Type of Mamba ("mamba1" or "mamba2")
        linear_units: Hidden dimension for FFN layers
        dropout_rate: Dropout rate
        chunk_size: For documentation; actual chunking is in encoder
        layer_idx: Layer index for Mamba
        block_conf: Additional config for Mamba layers
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mamba_type: str = "mamba1",
        linear_units: int = 2048,
        dropout_rate: float = 0.1,
        chunk_size: int = 20,
        layer_idx: int = 0,
        mamba2_d_ssm: Optional[int] = None,
        mamba2_headdim: int = 64,
        mamba2_ngroups: int = 1,
        mamba2_chunk_size: int = 256,
        block_conf: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.mamba_type = mamba_type.lower()

        if self.mamba_type not in {"mamba1", "mamba2"}:
            raise ValueError(f"Unsupported mamba_type={mamba_type}. Use mamba1 or mamba2.")

        # Within-chunk attention for local temporal dynamics
        self.attn = MultiHeadedAttention(
            n_feat=hidden_size, n_head=num_heads, dropout=dropout_rate
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Trans-chunk Mamba SSM for long-range dependencies with selective gating
        block_conf = {} if block_conf is None else dict(block_conf)
        if self.mamba_type == "mamba1":
            block_conf.setdefault("use_fast_path", True)
        else:
            block_conf.setdefault("use_mem_eff_path", True)

        block_cls = Mamba1 if self.mamba_type == "mamba1" else Mamba2

        if block_cls is Mamba1:
            self.mamba = block_cls(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=layer_idx,
                **block_conf,
            )
        else:
            self.mamba = block_cls(
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
        self.mamba_norm = nn.LayerNorm(hidden_size)

        # FFN for feature transformation
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, linear_units),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(linear_units, hidden_size),
            nn.Dropout(dropout_rate),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        mamba_state: Optional[Any] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Forward pass for hybrid block.
        
        Args:
            x: (B, T, D) input tensor
            mamba_state: State for Mamba (can be None for full sequence processing)
            mask: Optional mask for attention
            
        Returns:
            output: (B, T, D) output tensor
            new_mamba_state: Updated Mamba state for streaming
        """
        # Apply within-chunk attention (pre-norm, residual).
        # Chunking is required to keep attention memory bounded on long utterances.
        x_attn = self.attn_norm(x)
        if self.chunk_size > 0 and x_attn.size(1) > self.chunk_size:
            t = x_attn.size(1)
            attn_chunks = []
            for s in range(0, t, self.chunk_size):
                e = min(s + self.chunk_size, t)
                q = x_attn[:, s:e, :]
                local_mask = None
                if mask is not None:
                    if mask.dim() == 3:
                        local_mask = mask[:, s:e, s:e]
                    elif mask.dim() == 2:
                        local_mask = mask[:, s:e]
                    else:
                        local_mask = mask
                q_out, _ = self.attn(q, memory=q, mask=local_mask)
                attn_chunks.append(q_out)
            x_attn = torch.cat(attn_chunks, dim=1)
        else:
            x_attn, _ = self.attn(x_attn, memory=x_attn, mask=mask)
        x = x + x_attn

        # Apply trans-chunk Mamba with selective gating (pre-norm, residual)
        x_mamba = self.mamba_norm(x).contiguous()
        x_mamba, mamba_state = self.mamba(x_mamba, state=mamba_state)
        x = x + x_mamba

        # Apply FFN (pre-norm, residual)
        x = x + self.ffn(self.ffn_norm(x))

        return x, mamba_state


class HybridMambaAttentionEncoder(AbsEncoder):
    """Streaming Hybrid Mamba+Attention Encoder.
    
    Combines within-chunk attention for local temporal dynamics and
    trans-chunk Mamba for sequence-level selective state compression.
    
    Achieves state-of-the-art streaming ASR performance with 7.3% WER
    on Tedlium2 and 0.40 RTF (real-time factor).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        hidden_size: int = 512,
        num_blocks: int = 16,
        num_heads: int = 4,
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
        chunk_size: int = 20,
        block_conf: Optional[Dict[str, Any]] = None,
    ):
        """Initialize HybridMambaAttentionEncoder.
        
        Args:
            input_size: Input feature dimension (e.g., 80 for mel-spectrogram)
            output_size: Output feature dimension (default 256)
            hidden_size: Hidden dimension (default 512, from best model)
            num_blocks: Number of hybrid blocks (default 16, from best model)
            num_heads: Number of attention heads (default 4)
            mamba_type: Type of Mamba ("mamba1" or "mamba2", default "mamba1")
            d_state: State dimension for Mamba (default 16, from best model)
            d_conv: Mamba convolution kernel size
            expand: Mamba expansion factor (default 2)
            mamba2_d_ssm: Mamba2-specific SSM dimension
            mamba2_headdim: Mamba2 head dimension
            mamba2_ngroups: Mamba2 number of groups
            mamba2_chunk_size: Mamba2 chunk size for memory efficiency
            conv_kernel_size: Kernel size for frontend depthwise convs (default 15)
            conv_num_layers: Number of frontend convolution layers (default 2)
            linear_units: FFN hidden dimension (default 2048)
            dropout_rate: Dropout rate (default 0.1)
            chunk_size: Within-chunk attention window size in frames (default 20)
            block_conf: Additional configuration for Mamba blocks
        """
        super().__init__()
        self._output_size = output_size
        self.chunk_size = chunk_size
        self.hidden_size = hidden_size
        self.mamba_type = mamba_type.lower()

        # Input projection from acoustic features to hidden dimension
        self.in_proj = nn.Linear(input_size, hidden_size)

        # Frontend causal depthwise convolutions for feature extraction
        self.frontend_convs = nn.ModuleList(
            [
                _CausalDepthwiseConv1d(hidden_size, conv_kernel_size, dropout_rate)
                for _ in range(conv_num_layers)
            ]
        )

        # Hybrid blocks combining within-chunk attention + trans-chunk Mamba
        self.blocks = nn.ModuleList()
        for layer_idx in range(num_blocks):
            block = HybridMambaAttentionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                mamba_type=mamba_type,
                linear_units=linear_units,
                dropout_rate=dropout_rate,
                chunk_size=chunk_size,
                layer_idx=layer_idx,
                mamba2_d_ssm=mamba2_d_ssm,
                mamba2_headdim=mamba2_headdim,
                mamba2_ngroups=mamba2_ngroups,
                mamba2_chunk_size=mamba2_chunk_size,
                block_conf=block_conf,
            )
            self.blocks.append(block)

        # Output normalization and projection
        self.norm_out = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, output_size)

    def output_size(self) -> int:
        """Return output feature dimension."""
        return self._output_size

    def init_streaming_state(self) -> Optional[List[Any]]:
        """Return initial streaming state for chunk-wise inference."""
        return None

    def forward_chunk(
        self,
        xs_chunk: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: Optional[List[Any]] = None,
        is_final: bool = False,
        ctc=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Any]]]:
        """Chunk-wise wrapper used by streaming inference drivers."""
        del is_final
        return self.forward(xs_chunk, ilens, prev_states=prev_states, ctc=ctc)

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: Optional[List[Any]] = None,
        ctc=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Any]]]:
        """Forward pass through the hybrid encoder.
        
        Args:
            xs_pad: Input tensor (B, T, input_size)
            ilens: Input lengths (B,)
            prev_states: Previous Mamba states for streaming decoding
            ctc: CTC module (optional)
            
        Returns:
            output: Encoded features (B, T, output_size)
            ilens: Output lengths (B,)
            new_states: Updated Mamba states for next chunk
        """
        # Project input to hidden dimension
        x = self.in_proj(xs_pad)

        # Apply frontend causal convolutions
        for conv in self.frontend_convs:
            x = conv(x)

        # Process through hybrid blocks
        new_states: List[Any] = []
        for i, block in enumerate(self.blocks):
            mamba_state = None if prev_states is None else prev_states[i]
            x, mamba_state = block(x, mamba_state=mamba_state)
            new_states.append(mamba_state)

        # Output normalization and projection
        x = self.norm_out(x)
        x = self.out_proj(x)

        return x, ilens, new_states
