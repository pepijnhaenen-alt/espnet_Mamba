"""Decoder definition."""

import copy
from typing import Any, List, Tuple

import torch
from typeguard import typechecked

try:
    from mamba_ssm.utils.generation import InferenceParams
except ImportError:
    InferenceParams = None

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.state_spaces.model import SequenceModel
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.legacy.nets.scorer_interface import BatchScorerInterface


class S4Decoder(AbsDecoder, BatchScorerInterface):
    """S4 decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of hidden vector
        input_layer: input layer type
        dropinp: input dropout
        dropout: dropout parameter applied on every residual and every layer
        prenorm: pre-norm vs. post-norm
        n_layers: number of layers
        transposed: transpose inputs so each layer receives (batch, dim, length)
        tie_dropout: tie dropout mask across sequence like nn.Dropout1d/nn.Dropout2d
        n_repeat: each layer is repeated n times per stage before applying pooling
        layer: layer config, must be specified
        residual: residual config
        norm: normalization config (e.g. layer vs batch)
        pool: config for pooling layer per stage
        track_norms: log norms of each layer output
        drop_path: drop rate for stochastic depth
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        input_layer: str = "embed",
        dropinp: float = 0.0,
        dropout: float = 0.25,
        prenorm: bool = True,
        n_layers: int = 16,
        transposed: bool = False,
        tie_dropout: bool = False,
        n_repeat=1,
        layer=None,
        residual=None,
        norm=None,
        pool=None,
        track_norms=True,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.d_model = encoder_output_size
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.odim = vocab_size
        self.dropout = dropout

        if input_layer == "embed":
            self.embed = torch.nn.Embedding(vocab_size, self.d_model)
        else:
            raise NotImplementedError
        self.dropout_emb = torch.nn.Dropout(p=dropout)

        self.decoder = SequenceModel(
            self.d_model,
            n_layers=n_layers,
            transposed=transposed,
            dropout=dropout,
            tie_dropout=tie_dropout,
            prenorm=prenorm,
            n_repeat=n_repeat,
            layer=layer,
            residual=residual,
            norm=norm,
            pool=pool,
            track_norms=track_norms,
            dropinp=dropinp,
            drop_path=drop_path,
        )

        self.output = torch.nn.Linear(self.d_model, vocab_size)

    def init_state(self, x: torch.Tensor):
        """Initialize state."""
        return self.decoder.default_state(1, device=x.device)

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        state=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )

        emb = self.embed(ys_in_pad)
        z, state = self.decoder(
            emb,
            state=state,
            memory=memory,
            lengths=ys_in_lens,
            mask=memory_mask,
        )

        decoded = self.output(z)
        return decoded, ys_in_lens

    def score(self, ys, state, x):
        raise NotImplementedError

    def _merge_layer_state(self, states: List[Any]) -> Any:
        """Merge per-hypothesis states into a single batched state."""
        if len(states) == 0:
            return None

        state0 = states[0]
        if state0 is None:
            return None

        if torch.is_tensor(state0):
            return torch.cat(states, dim=0)

        if isinstance(state0, tuple):
            return tuple(
                self._merge_layer_state([s[i] for s in states])
                for i in range(len(state0))
            )

        if isinstance(state0, list):
            return [
                self._merge_layer_state([s[i] for s in states])
                for i in range(len(state0))
            ]

        if isinstance(state0, dict):
            return {
                k: self._merge_layer_state([s[k] for s in states]) for k in state0
            }

        if InferenceParams is not None and isinstance(state0, InferenceParams):
            merged = InferenceParams(
                max_seqlen=state0.max_seqlen,
                max_batch_size=len(states),
            )
            merged.seqlen_offset = state0.seqlen_offset
            if hasattr(state0, "batch_size_offset"):
                merged.batch_size_offset = state0.batch_size_offset

            lengths = [
                s.lengths_per_sample
                for s in states
                if getattr(s, "lengths_per_sample", None) is not None
            ]
            if len(lengths) > 0:
                merged.lengths_per_sample = torch.cat(lengths, dim=0)

            merged.key_value_memory_dict = {
                k: self._merge_layer_state([s.key_value_memory_dict[k] for s in states])
                for k in state0.key_value_memory_dict
            }
            return merged

        return copy.deepcopy(state0)

    def _split_layer_state(self, state: Any, index: int) -> Any:
        """Extract a single-hypothesis state from a batched state."""
        if state is None:
            return None

        if torch.is_tensor(state):
            return state[index : index + 1]

        if isinstance(state, tuple):
            return tuple(self._split_layer_state(s, index) for s in state)

        if isinstance(state, list):
            return [self._split_layer_state(s, index) for s in state]

        if isinstance(state, dict):
            return {k: self._split_layer_state(v, index) for k, v in state.items()}

        if InferenceParams is not None and isinstance(state, InferenceParams):
            split = InferenceParams(max_seqlen=state.max_seqlen, max_batch_size=1)
            split.seqlen_offset = state.seqlen_offset
            if hasattr(state, "batch_size_offset"):
                split.batch_size_offset = state.batch_size_offset
            if getattr(state, "lengths_per_sample", None) is not None:
                split.lengths_per_sample = state.lengths_per_sample[index : index + 1]
            split.key_value_memory_dict = {
                k: self._split_layer_state(v, index)
                for k, v in state.key_value_memory_dict.items()
            }
            return split

        return copy.deepcopy(state)

    def _merge_batch_states(self, states: List[Any]) -> Any:
        """Convert beam-search list states into decoder step batched format."""
        if states is None or len(states) == 0:
            return None
        if states[0] is None:
            return None

        if isinstance(states[0], list):
            n_layers = len(states[0])
            return [
                self._merge_layer_state([state[layer_idx] for state in states])
                for layer_idx in range(n_layers)
            ]

        return states

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        n_batch = len(ys)
        ys = self.embed(ys[:, -1:])

        states = self._merge_batch_states(states)

        assert ys.size(1) == 1, ys.shape
        ys = ys.squeeze(1)

        ys, states = self.decoder.step(ys, state=states, memory=xs)
        logp = self.output(ys).log_softmax(dim=-1)

        states_list = [
            [self._split_layer_state(layer_state, b) for layer_state in states]
            for b in range(n_batch)
        ]

        return logp, states_list
