### Fixtures and convolution unit tests

"""
Tests for the streaming causal convolution.

These tests only exercise _CausalDepthwiseConv1d.
No Mamba blocks are instantiated.

Run:

    pytest tests/test_mamba_encoder_streaming.py -v
"""

import torch
import pytest
import torch.nn.functional as F

from espnet2.asr.encoder.mamba_encoder import _CausalDepthwiseConv1d, MambaEncoder


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture(scope="function")
def seed():
    torch.manual_seed(0)


@pytest.fixture(
    params=[
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA unavailable",
            ),
        ),
    ]
)
def device(request):
    return torch.device(request.param)


@pytest.fixture
def channels():
    return 32


@pytest.fixture
def kernel():
    return 15


@pytest.fixture
def dropout():
    return 0.0


@pytest.fixture
def conv(channels, kernel, dropout, device):
    layer = _CausalDepthwiseConv1d(
        channels=channels,
        kernel_size=kernel,
        dropout_rate=dropout,
    )
    layer.to(device)
    layer.eval()
    return layer


# -------------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------------


def offline_reference(conv, x):
    """
    Computes the reference convolution using explicit left padding.

    This should exactly match the offline path of the module.
    """

    x = x.transpose(1, 2)
    x = F.pad(x, (conv.kernel_size - 1, 0))
    y = conv.conv(x)
    y = conv.act(y)
    y = conv.dropout(y)
    return y.transpose(1, 2)


# -------------------------------------------------------------------------
# Basic construction
# -------------------------------------------------------------------------


def test_conv_constructs(conv):
    assert isinstance(conv.conv, torch.nn.Conv1d)
    assert conv.kernel_size == 15


def test_dropout_disabled(conv):
    assert conv.dropout.p == 0.0


# -------------------------------------------------------------------------
# Offline behaviour
# -------------------------------------------------------------------------


def test_output_shape(conv, device):

    x = torch.randn(2, 50, 32, device=device)

    y, cache = conv(x)

    assert y.shape == x.shape

    assert cache.shape == (
        2,
        32,
        conv.kernel_size - 1,
    )


def test_offline_matches_reference(conv, device):

    x = torch.randn(2, 80, 32, device=device)

    y, _ = conv(x)

    ref = offline_reference(conv, x)

    torch.testing.assert_close(
        y,
        ref,
        atol=1e-6,
        rtol=1e-6,
    )


# -------------------------------------------------------------------------
# Cache
# -------------------------------------------------------------------------


def test_cache_size(conv, device):

    x = torch.randn(3, 25, 32, device=device)

    _, cache = conv(x)

    assert cache.shape == (
        3,
        32,
        conv.kernel_size - 1,
    )


def test_cache_contains_last_frames(conv, device):

    T = 40

    x = torch.randn(1, T, 32, device=device)

    _, cache = conv(x)

    expected = (
        x.transpose(1, 2)[
            :, :, -(conv.kernel_size - 1):
        ]
    )

    torch.testing.assert_close(cache, expected)


def test_cache_not_modified(conv, device):

    x = torch.randn(2, 20, 32, device=device)

    _, cache1 = conv(x)

    _, cache2 = conv(x)

    torch.testing.assert_close(cache1, cache2)


# -------------------------------------------------------------------------
# Streaming path
# -------------------------------------------------------------------------


def test_streaming_accepts_cache(conv, device):

    x = torch.randn(2, 30, 32, device=device)

    _, cache = conv(x)

    y, new_cache = conv(x, cache)

    assert y.shape == x.shape

    assert new_cache.shape == cache.shape


def test_cache_updates(conv, device):

    x1 = torch.randn(1, 20, 32, device=device)
    x2 = torch.randn(1, 20, 32, device=device)

    _, cache = conv(x1)

    _, new_cache = conv(x2, cache)

    expected = (
        x2.transpose(1, 2)[
            :, :, -(conv.kernel_size - 1):
        ]
    )

    torch.testing.assert_close(
        new_cache,
        expected,
    )


# -------------------------------------------------------------------------
# Edge cases
# -------------------------------------------------------------------------


def test_single_frame(conv, device):

    x = torch.randn(1, 1, 32, device=device)

    y, cache = conv(x)

    assert y.shape == x.shape

    assert cache.shape[-1] == conv.kernel_size - 1


def test_chunk_smaller_than_kernel(conv, device):

    x = torch.randn(1, 4, 32, device=device)

    y, cache = conv(x)

    assert y.shape == x.shape

    assert cache.shape[-1] == conv.kernel_size - 1


def test_large_chunk(conv, device):

    x = torch.randn(2, 500, 32, device=device)

    y, cache = conv(x)

    assert y.shape == x.shape

    assert cache.shape[-1] == conv.kernel_size - 1


# -------------------------------------------------------------------------
# Gradients
# -------------------------------------------------------------------------


def test_backward(conv, device):

    x = torch.randn(
        2,
        40,
        32,
        device=device,
        requires_grad=True,
    )

    y, _ = conv(x)

    loss = y.mean()

    loss.backward()

    assert x.grad is not None

    assert torch.isfinite(x.grad).all()

    assert conv.conv.weight.grad is not None

    assert torch.isfinite(conv.conv.weight.grad).all()


# -------------------------------------------------------------------------
# Determinism
# -------------------------------------------------------------------------


def test_deterministic(seed, conv, device):

    x = torch.randn(2, 40, 32, device=device)

    y1, _ = conv(x)

    torch.manual_seed(0)

    y2, _ = conv(x)

    torch.testing.assert_close(y1, y2)


# Encoder integration tests
"""
Encoder integration tests.

These tests verify the complete MambaEncoder interface.
They do NOT compare streaming and offline outputs yet;
those belong to the regression tests.

Run:

    pytest tests/test_mamba_encoder_streaming.py -v
"""

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture(scope="function")
def seed():
    torch.manual_seed(0)


@pytest.fixture(params=["mamba1", "mamba2"])
def encoder(request, device):
    if request.param == "mamba1":
        model = MambaEncoder(
            input_size=80,
            output_size=256,
            hidden_size=64,
            num_blocks=2,
            d_state=8,
            d_conv=4,
            expand=2,
            conv_kernel_size=7,
            conv_num_layers=2,
            linear_units=128,
            dropout_rate=0.0,
        )
        model.to(device)
        model.eval()
        return model
    elif request.param == "mamba2":
        model = MambaEncoder(
            input_size=80,
            output_size=256,
            hidden_size=64,
            num_blocks=2,
            d_state=8,
            d_conv=4,
            expand=2,
            conv_kernel_size=7,
            conv_num_layers=2,
            linear_units=128,
            dropout_rate=0.0,
            mamba_type=request.param
        )
        model.to(device)
        model.eval()
        return model


# -------------------------------------------------------------------------
# Construction
# -------------------------------------------------------------------------


def test_output_size(encoder):
    assert encoder.output_size() == 256


def test_num_blocks(encoder):
    assert len(encoder.blocks) == 2


def test_num_frontend_convs(encoder):
    assert len(encoder.frontend_convs) == 2


# -------------------------------------------------------------------------
# Streaming state initialization
# -------------------------------------------------------------------------


def test_init_streaming_state_structure(encoder, device):

    state = encoder.init_streaming_state(
        batch_size=2,
        device=device,
    )

    assert isinstance(state, dict)

    assert "mamba" in state
    assert "conv" in state

    assert len(state["mamba"]) == len(encoder.blocks)
    assert len(state["conv"]) == len(encoder.frontend_convs)


def test_conv_state_shapes(encoder, device):

    state = encoder.init_streaming_state(
        batch_size=3,
        device=device,
    )

    for cache in state["conv"]:

        assert cache.shape[0] == 3
        assert cache.shape[1] == encoder.hidden_size
        assert cache.shape[2] == (
            encoder.conv_kernel_size - 1
        )


def test_conv_state_zero_initialized(encoder, device):

    state = encoder.init_streaming_state()

    for cache in state["conv"]:
        assert torch.all(cache == 0)


# -------------------------------------------------------------------------
# Offline forward
# -------------------------------------------------------------------------


def test_forward_shape(encoder, device):

    x = torch.randn(
        2,
        100,
        80,
        device=device,
    )

    ilens = torch.tensor([100, 90], device=device)

    y, olens, state = encoder(
        x,
        ilens,
    )

    assert y.shape == (
        2,
        100,
        encoder.output_size(),
    )

    assert torch.equal(
        olens,
        ilens,
    )

    assert state is None


def test_forward_single_frame(encoder, device):

    x = torch.randn(
        1,
        1,
        80,
        device=device,
    )

    ilens = torch.tensor([1])

    y, _, _ = encoder(
        x,
        ilens,
    )

    assert y.shape[1] == 1


def test_forward_empty_batch(encoder, device):

    x = torch.randn(
        0,
        10,
        80,
        device=device,
    )

    ilens = torch.empty(
        0,
        dtype=torch.long,
        device=device,
    )

    y, olens, _ = encoder(
        x,
        ilens,
    )

    assert y.shape[0] == 0
    assert len(olens) == 0


# -------------------------------------------------------------------------
# Streaming forward
# -------------------------------------------------------------------------


def test_forward_chunk_returns_state(
    encoder,
    device,
):

    x = torch.randn(
        1,
        20,
        80,
        device=device,
    )

    ilens = torch.tensor([20])

    state = encoder.init_streaming_state()

    y, olens, new_state = encoder.forward_chunk(
        x,
        ilens,
        prev_states=state,
    )

    assert y.shape[1] == 20

    assert new_state is not None

    assert "mamba" in new_state
    assert "conv" in new_state


def test_forward_chunk_preserves_state_lengths(
    encoder,
    device,
):

    x = torch.randn(
        1,
        20,
        80,
        device=device,
    )

    ilens = torch.tensor([20])

    state = encoder.init_streaming_state()

    _, _, state = encoder.forward_chunk(
        x,
        ilens,
        prev_states=state,
    )

    assert len(state["mamba"]) == len(encoder.blocks)

    assert len(state["conv"]) == len(
        encoder.frontend_convs
    )


# -------------------------------------------------------------------------
# State evolution
# -------------------------------------------------------------------------


def test_state_changes_after_chunk(
    encoder,
    device,
):

    x = torch.randn(
        1,
        20,
        80,
        device=device,
    )

    ilens = torch.tensor([20])

    state = encoder.init_streaming_state()

    old_conv = [
        c.clone()
        for c in state["conv"]
    ]

    _, _, state = encoder.forward_chunk(
        x,
        ilens,
        prev_states=state,
    )

    changed = False

    for old, new in zip(
        old_conv,
        state["conv"],
    ):
        if not torch.equal(old, new):
            changed = True

    assert changed


# -------------------------------------------------------------------------
# Gradient flow
# -------------------------------------------------------------------------


def test_backward(
    encoder,
    device,
):

    encoder.train()

    x = torch.randn(
        2,
        30,
        80,
        device=device,
        requires_grad=True,
    )

    ilens = torch.tensor([30, 30])

    y, _, _ = encoder(
        x,
        ilens,
    )

    loss = y.mean()

    loss.backward()

    assert x.grad is not None

    assert torch.isfinite(
        x.grad
    ).all()

    assert (
        encoder.in_proj.weight.grad
        is not None
    )

    assert torch.isfinite(
        encoder.in_proj.weight.grad
    ).all()


# -------------------------------------------------------------------------
# Batch sizes
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4],
)
def test_batch_sizes(
    encoder,
    device,
    batch_size,
):

    x = torch.randn(
        batch_size,
        40,
        80,
        device=device,
    )

    ilens = torch.full(
        (batch_size,),
        40,
        dtype=torch.long,
        device=device,
    )

    y, _, _ = encoder(
        x,
        ilens,
    )

    assert y.shape[0] == batch_size


# -------------------------------------------------------------------------
# Determinism
# -------------------------------------------------------------------------


def test_encoder_deterministic(
    encoder,
    device,
):

    torch.manual_seed(0)

    x = torch.randn(
        2,
        50,
        80,
        device=device,
    )

    ilens = torch.tensor([50, 50])

    y1, _, _ = encoder(
        x,
        ilens,
    )

    torch.manual_seed(0)

    y2, _, _ = encoder(
        x,
        ilens,
    )

    torch.testing.assert_close(
        y1,
        y2,
    )

def test_streaming_state_device(encoder, device):
    state = encoder.init_streaming_state(batch_size=2, device=device)

    for cache in state["conv"]:
        print(cache.device, device)
        assert cache.device == device

    for mamba_state in state["mamba"]:
        assert mamba_state.conv_state.device == device
        assert mamba_state.ssm_state.device == device

# Streaming equivalence and regression tests

def run_streaming(
    encoder,
    x,
    ilens,
    chunk_sizes,
):
    """
    Run encoder chunk-by-chunk and concatenate outputs.
    """

    state = encoder.init_streaming_state(
        batch_size=x.size(0),
        device=x.device,
    )

    outputs = []

    start = 0

    for chunk in chunk_sizes:

        end = min(start + chunk, x.size(1))

        xs = x[:, start:end, :]

        lens = torch.full(
            (x.size(0),),
            end - start,
            dtype=torch.long,
            device=x.device,
        )

        y, _, state = encoder.forward_chunk(
            xs,
            lens,
            prev_states=state,
            is_final=(end == x.size(1)),
        )

        outputs.append(y)

        start = end

        if end == x.size(1):
            break

    return torch.cat(outputs, dim=1)

def test_streaming_equals_offline_fixed_chunks(
    encoder,
    device,
):

    x = torch.randn(
        2,
        200,
        80,
        device=device,
    )

    ilens = torch.tensor(
        [200, 200],
        device=device,
    )

    offline, _, _ = encoder(
        x,
        ilens,
    )
    print("Offline:", offline.size())

    streamed = run_streaming(
        encoder,
        x,
        ilens,
        [40, 40, 40, 40, 40],
    )
    print("Online:", streamed.size())

    torch.testing.assert_close(
        offline,
        streamed,
        atol=1e-5,
        rtol=1e-5,
    )

def test_streaming_equals_offline_uneven(
    encoder,
    device,
):

    x = torch.randn(
        1,
        173,
        80,
        device=device,
    )

    ilens = torch.tensor([173])

    offline, _, _ = encoder(
        x,
        ilens,
    )

    streamed = run_streaming(
        encoder,
        x,
        ilens,
        [17, 31, 8, 56, 61],
    )

    torch.testing.assert_close(
        offline,
        streamed,
        atol=1e-5,
        rtol=1e-5,
    )

def test_random_chunk_sizes(
    encoder,
    device,
):

    torch.manual_seed(0)

    x = torch.randn(
        1,
        300,
        80,
        device=device,
    )

    ilens = torch.tensor([300])

    offline, _, _ = encoder(
        x,
        ilens,
    )

    remaining = 300

    chunks = []

    while remaining > 0:

        c = min(
            torch.randint(
                1,
                50,
                (1,),
            ).item(),
            remaining,
        )

        chunks.append(c)

        remaining -= c

    streamed = run_streaming(
        encoder,
        x,
        ilens,
        chunks,
    )

    torch.testing.assert_close(
        offline,
        streamed,
        atol=1e-5,
        rtol=1e-5,
    )

def test_frame_by_frame_streaming(
    encoder,
    device,
):

    x = torch.randn(
        1,
        100,
        80,
        device=device,
    )

    ilens = torch.tensor([100])

    offline, _, _ = encoder(
        x,
        ilens,
    )

    streamed = run_streaming(
        encoder,
        x,
        ilens,
        [1] * 100,
    )

    torch.testing.assert_close(
        offline,
        streamed,
        atol=1e-5,
        rtol=1e-5,
    )

@pytest.mark.parametrize(
    "chunk",
    [32, 64, 128],
)
def test_large_chunks(
    encoder,
    device,
    chunk,
):

    x = torch.randn(
        2,
        256,
        80,
        device=device,
    )

    ilens = torch.tensor(
        [256, 256],
        device=device,
    )

    offline, _, _ = encoder(
        x,
        ilens,
    )

    streamed = run_streaming(
        encoder,
        x,
        ilens,
        [chunk] * 20,
    )

    torch.testing.assert_close(
        offline,
        streamed,
        atol=1e-5,
        rtol=1e-5,
    )

def test_long_streaming(
    encoder,
    device,
):

    x = torch.randn(
        1,
        2000,
        80,
        device=device,
    )

    ilens = torch.tensor([2000])

    offline, _, _ = encoder(
        x,
        ilens,
    )

    streamed = run_streaming(
        encoder,
        x,
        ilens,
        [37] * 100,
    )

    torch.testing.assert_close(
        offline,
        streamed,
        atol=1e-5,
        rtol=1e-5,
    )

def test_seqlen_offset_increases(
    encoder,
    device,
):

    state = encoder.init_streaming_state()

    x = torch.randn(
        1,
        25,
        80,
        device=device,
    )

    ilens = torch.tensor([25])

    _, _, state = encoder.forward_chunk(
        x,
        ilens,
        prev_states=state,
    )

    for s in state["mamba"]:

        assert s.seqlen_offset == 25

def test_state_reset(
    encoder,
    device,
):

    state = encoder.init_streaming_state()

    x = torch.randn(
        1,
        20,
        80,
        device=device,
    )

    ilens = torch.tensor([20])

    _, _, state = encoder.forward_chunk(
        x,
        ilens,
        prev_states=state,
        is_final=True,
    )

    new_state = encoder.init_streaming_state()

    for c in new_state["conv"]:
        assert torch.all(c == 0)

def test_multiple_utterances(
    encoder,
    device,
):

    for _ in range(3):

        x = torch.randn(
            1,
            120,
            80,
            device=device,
        )

        ilens = torch.tensor([120])

        streamed = run_streaming(
            encoder,
            x,
            ilens,
            [30, 30, 30, 30],
        )

        assert streamed.shape[1] == 120

def test_streaming_deterministic(
    encoder,
    device,
):

    torch.manual_seed(42)

    x = torch.randn(
        1,
        150,
        80,
        device=device,
    )

    ilens = torch.tensor([150])

    y1 = run_streaming(
        encoder,
        x,
        ilens,
        [25] * 6,
    )

    torch.manual_seed(42)

    y2 = run_streaming(
        encoder,
        x,
        ilens,
        [25] * 6,
    )

    torch.testing.assert_close(
        y1,
        y2,
    )

def test_short_chunk_cache_regression(
    encoder,
    device,
):
    """
    Streaming with chunks much smaller than the convolution
    kernel should still match offline inference.

    This test catches incorrect cache updates.
    """

    torch.manual_seed(0)

    x = torch.randn(
        1,
        120,
        80,
        device=device,
    )

    ilens = torch.tensor([120], device=device)

    offline, _, _ = encoder(
        x,
        ilens,
    )

    streamed = run_streaming(
        encoder,
        x,
        ilens,
        [2] * 60,
    )

    torch.testing.assert_close(
        offline,
        streamed,
        atol=1e-5,
        rtol=1e-5,
    )

@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 5])
def test_short_chunk_cache_regression(
    encoder,
    device,
    chunk_size,
):

    x = torch.randn(1, 150, 80, device=device)
    ilens = torch.tensor([150], device=device)

    offline, _, _ = encoder(x, ilens)

    n_chunks = (150 + chunk_size - 1) // chunk_size
    chunks = [chunk_size] * n_chunks

    streamed = run_streaming(
        encoder,
        x,
        ilens,
        chunks,
    )

    torch.testing.assert_close(
        offline,
        streamed,
        atol=1e-5,
        rtol=1e-5,
    )

def test_chunk_boundary_invariance(
    encoder,
    device,
):
    """
    Different chunk boundaries should produce identical outputs.
    """

    torch.manual_seed(1)

    x = torch.randn(
        1,
        240,
        80,
        device=device,
    )

    ilens = torch.tensor([240], device=device)

    y1 = run_streaming(
        encoder,
        x,
        ilens,
        [60, 60, 60, 60],
    )

    y2 = run_streaming(
        encoder,
        x,
        ilens,
        [40, 40, 40, 40, 40, 40],
    )

    y3 = run_streaming(
        encoder,
        x,
        ilens,
        [17, 51, 33, 74, 65],
    )

    torch.testing.assert_close(
        y1,
        y2,
        atol=1e-5,
        rtol=1e-5,
    )

    torch.testing.assert_close(
        y1,
        y3,
        atol=1e-5,
        rtol=1e-5,
    )

@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4],
)
def test_streaming_batch_sizes(
    encoder,
    device,
    batch_size,
):
    """
    Offline and streaming inference should agree for
    multiple batch sizes.
    """

    torch.manual_seed(2)

    x = torch.randn(
        batch_size,
        180,
        80,
        device=device,
    )

    ilens = torch.full(
        (batch_size,),
        180,
        dtype=torch.long,
        device=device,
    )

    offline, _, _ = encoder(
        x,
        ilens,
    )

    streamed = run_streaming(
        encoder,
        x,
        ilens,
        [30] * 6,
    )

    torch.testing.assert_close(
        offline,
        streamed,
        atol=1e-5,
        rtol=1e-5,
    )

def test_random_chunk_boundary_invariance(
    encoder,
    device,
):
    """
    Random chunk boundaries should not affect the encoder output.
    """

    torch.manual_seed(3)

    x = torch.randn(
        1,
        300,
        80,
        device=device,
    )

    ilens = torch.tensor([300], device=device)

    reference = run_streaming(
        encoder,
        x,
        ilens,
        [50] * 6,
    )

    for _ in range(10):

        remaining = 300
        chunks = []

        while remaining > 0:
            c = min(
                torch.randint(1, 60, (1,)).item(),
                remaining,
            )
            chunks.append(c)
            remaining -= c

        candidate = run_streaming(
            encoder,
            x,
            ilens,
            chunks,
        )

        torch.testing.assert_close(
            reference,
            candidate,
            atol=1e-5,
            rtol=1e-5,
        )

def test_streaming_state_independence(
    encoder,
    device,
):
    """
    Two identical utterances processed with fresh streaming
    states should produce identical outputs.
    """

    torch.manual_seed(4)

    x = torch.randn(
        1,
        160,
        80,
        device=device,
    )

    ilens = torch.tensor([160], device=device)

    y1 = run_streaming(
        encoder,
        x,
        ilens,
        [40, 40, 40, 40],
    )

    y2 = run_streaming(
        encoder,
        x,
        ilens,
        [40, 40, 40, 40],
    )

    torch.testing.assert_close(
        y1,
        y2,
        atol=1e-5,
        rtol=1e-5,
    )

# Stress tests and GPU tests