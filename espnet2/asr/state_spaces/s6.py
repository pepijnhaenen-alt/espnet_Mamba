# This code implements Mamba1 and Mamba2 blocks for ESPnet
# Based on the mamba_ssm library

import math
from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2 as Mamba2SSM
from mamba_ssm.utils.generation import InferenceParams

from espnet2.asr.state_spaces.base import SequenceModule


class Mamba1(SequenceModule):
    """Mamba1 block for ESPnet, based on the original Mamba architecture."""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            use_fast_path=use_fast_path,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )
        # Set layer_idx for state management
        self.layer_idx = layer_idx

    @property
    def d_output(self):
        return self.d_model

    def forward(self, x, state=None, **kwargs):
        """Forward pass.

        x: (B, L, D) input tensor
        Returns: (B, L, D) output tensor, state
        """
        # print("forward")
        # Mamba expects (B, L, D)
        logger = logging.getLogger(__name__)
        logger.debug("Mamba1.forward: fused call, state present=%s", state is not None)

        if state is None:
            # # Initialize state if not provided
            # state = self.default_state(x.shape[0], device=x.device)
            # print("state is None")
            y = self.mamba(x)
            return y, None

        # try:
        #     y = self.mamba(x, inference_params = state)
        #     state.seqlen_offset += x.size(1)
        #     return y, state
        # except:
        y_list = []
        for i in range(x.size()[1]):
            xi = x[:,i,:]
            y, state = self.step(xi, state)
            y = y.unsqueeze(1)
            y_list.append(y)
        return torch.cat(y_list, dim=1), state
        
        # print(state.seqlen_offset)

        # y_list = []
        # for t in range(x.shape[1]):
        #     y_t, inference_params = self.step(x[:, t, :], inference_params) # (B, L, D) -> (B, D)
        #     y_list.append(y_t.unsqueeze(1)) 
        # y = torch.cat(y_list, dim=1) # [(B, D)] -> (B, L, D)

        

    def step(self, x, state, **kwargs):
        """Step function for recurrent inference.

        x: (B, D) input at current timestep
        state: InferenceParams object containing the state
        Returns: (B, D) output, new state
        """
        if state is None:
            # Initialize state if not provided
            state = self.default_state(x.shape[0], device=x.device)
        
        # Use layer_idx if set, otherwise assert
        assert self.layer_idx is not None
        layer_key = self.layer_idx
        
        # Get states from cache
        conv_state, ssm_state = state.key_value_memory_dict[layer_key]
        
        # mamba_ssm Mamba.step expects hidden_states with a time axis: (B, 1, D).
        x_step = x.unsqueeze(1) if x.dim() == 2 else x
        y, conv_state, ssm_state = self.mamba.step(x_step, conv_state, ssm_state)
        if y.dim() == 3 and y.size(1) == 1:
            y = y.squeeze(1)
        
        # Update the inference params
        state.key_value_memory_dict[layer_key] = (conv_state, ssm_state)
        state.seqlen_offset += 1
        
        return y, state

    def default_state(self, batch_size:int=1, device=None, **kwargs):
        """Default state for initialization."""
        if device is None:
            device = next(self.parameters()).device
        
        max_seqlen = 1  # Will be updated as we step
        max_batch_size= batch_size
        # Create InferenceParams with allocated cache for this layer
        inference_params = InferenceParams(
            max_seqlen = max_seqlen,  # Will be updated as we step
            max_batch_size= max_batch_size,
        )
        
        # Use layer_idx if set, otherwise assert
        assert self.layer_idx is not None
        layer_key = self.layer_idx
        
        # Allocate cache for this layer
        conv_state, ssm_state = self.mamba.allocate_inference_cache(
            batch_size=max_batch_size,
            max_seqlen=max_seqlen,
            dtype=self.mamba.conv1d.weight.dtype,
            device=device,
        )
        
        # Store in the inference params
        inference_params.key_value_memory_dict[layer_key] = (conv_state, ssm_state)
        
        return inference_params


class Mamba2(SequenceModule):
    """Mamba2 block for ESPnet, based on the improved Mamba2 architecture."""

    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.mamba2 = Mamba2SSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            conv_init=conv_init,
            expand=expand,
            headdim=headdim,
            d_ssm=d_ssm,
            ngroups=ngroups,
            A_init_range=A_init_range,
            D_has_hdim=D_has_hdim,
            rmsnorm=rmsnorm,
            norm_before_gate=norm_before_gate,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            dt_limit=dt_limit,
            bias=bias,
            conv_bias=conv_bias,
            chunk_size=chunk_size,
            use_mem_eff_path=use_mem_eff_path,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )
        # Set layer_idx for state management
        self.layer_idx = layer_idx

    @property
    def d_output(self):
        return self.d_model

    def forward(self, x, state=None):
        print("Wrapper input:")
        print("  shape      :", x.shape)
        print("  stride     :", x.stride())
        print("  contiguous :", x.is_contiguous())
        if x.stride(-1) != 1:
            x = x.contiguous()
        # x = x.contiguous()

        print("After contiguous:")
        print("  stride     :", x.stride())

        if x.shape[-1] != self.d_model:
            x = x.transpose(1,2)

        if state is None:
            return self.mamba2(x), None
        
        assert self.layer_idx is not None

        y = self.mamba2(
            x,
            inference_params=state,
        )

        state.seqlen_offset += x.size(1)

        return y, state
        

    def step(self, x, state):
        if x.stride(-1) != 1:
            x = x.contiguous()
        # x = x.contiguous()

        if x.shape[-1] != self.d_model:
            x = x.transpose(1,2)

        if x.dim() == 2:
            x = x.unsqueeze(1)

        if self.layer_idx not in state.key_value_memory_dict:
            conv_state, ssm_state = \
                self.mamba2.allocate_inference_cache(
                    batch_size=1,
                    max_seqlen=1,
                    dtype=self.mamba2.in_proj.weight.dtype,
                    device=self.device,
                    )
            state.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
            )

        y, conv_state, ssm_state = self.mamba2.step(
            x,
            conv_state,
            ssm_state,
        )

        state.key_value_memory_dict[self.layer_idx] = (
            conv_state,
            ssm_state,
        )

        # state.seqlen_offset += 1

        if y.dim() == 3:
            y = y.squeeze(1)

        return y, state

    def default_state(self, batch_size=1, device=None, **kwargs):
        """Default state for initialization."""
        if device is None:
            device = next(self.parameters()).device
        
        max_seqlen = kwargs.get("max_seqlen", 4096) # Will be updated as we step
        # Create InferenceParams with allocated cache for this layer
        inference_params = InferenceParams(
            max_seqlen = max_seqlen,  # Will be updated as we step
            max_batch_size= batch_size,
        )
        
        # Use layer_idx if set, otherwise assert
        assert self.layer_idx is not None
        layer_key = self.layer_idx
        
        # Allocate cache for this layer
        conv_state, ssm_state = self.mamba2.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=max_seqlen,
            dtype=self.mamba2.in_proj.weight.dtype,
            device=device,
        )

        # Store in the inference params
        inference_params.key_value_memory_dict[layer_key] = (conv_state, ssm_state)
        
        return inference_params