# tests/test_mamba_encoder.py

import copy

import pytest
import torch

from espnet2.asr.encoder.mamba_encoder import MambaEncoder

print(torch.cuda.is_available())

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 4
TIME = 32
INPUT_DIM = 80


def make_encoder():
    encoder = MambaEncoder(
        input_size=INPUT_DIM,
        output_size=64,
        num_blocks=2,
        dropout_rate=0.0,
    )
    return encoder.to(device)


def make_inputs():
    xs = torch.randn(BATCH_SIZE, TIME, INPUT_DIM).to(device)
    lengths = torch.tensor([32, 28, 20, 16], dtype=torch.long).to(device)
    return xs, lengths


def test_encoder_constructs():
    """
    Verify constructor works and required attributes exist.
    """
    encoder = make_encoder()

    assert encoder is not None, (
        "Failed to construct MambaEncoder."
    )

    assert callable(getattr(encoder, "forward", None)), (
        "MambaEncoder does not implement forward()."
    )


def test_forward_shapes():
    """
    Verify output dimensions are correct.
    """
    encoder = make_encoder()

    xs, lengths = make_inputs()

    ys, out_lengths, _ = encoder(xs, lengths)

    assert ys.shape[0] == BATCH_SIZE, (
        f"Batch dimension mismatch. "
        f"Expected {BATCH_SIZE}, got {ys.shape[0]}"
    )

    assert ys.shape[1] == TIME, (
        f"Time dimension mismatch. "
        f"Expected {TIME}, got {ys.shape[1]}"
    )

    assert ys.shape[2] == 64, (
        f"Feature dimension mismatch. "
        f"Expected 64, got {ys.shape[2]}"
    )

    assert torch.equal(out_lengths, lengths), (
        f"Output lengths changed unexpectedly.\n"
        f"Expected: {lengths.tolist()}\n"
        f"Got: {out_lengths.tolist()}"
    )


def test_forward_no_nan():
    """
    Ensure numerical stability.
    """
    encoder = make_encoder()

    xs, lengths = make_inputs()

    ys, _, _ = encoder(xs, lengths)

    assert not torch.isnan(ys).any(), (
        "NaN values detected in encoder output."
    )

    assert not torch.isinf(ys).any(), (
        "Inf values detected in encoder output."
    )


def test_backward_pass():
    """
    Verify gradients flow through all trainable parameters.
    """
    encoder = make_encoder()

    xs, lengths = make_inputs()

    ys, _, _ = encoder(xs, lengths)

    loss = ys.pow(2).mean()
    loss.backward()

    missing_grads = []

    for name, param in encoder.named_parameters():
        if param.requires_grad and param.grad is None:
            missing_grads.append(name)

    assert not missing_grads, (
        "The following parameters did not receive gradients:\n"
        + "\n".join(missing_grads)
    )


def test_parameters_update():
    """
    Verify optimizer step updates parameters.
    """
    encoder = make_encoder()

    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=1e-3,
    )

    before = {
        name: p.detach().clone()
        for name, p in encoder.named_parameters()
    }

    xs, lengths = make_inputs()

    ys, _, _ = encoder(xs, lengths)

    loss = ys.square().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    updated = False

    for name, p in encoder.named_parameters():
        if not torch.allclose(before[name], p):
            updated = True
            break

    assert updated, (
        "Optimizer step completed but no parameters changed."
    )


def test_tiny_overfit():
    """
    Strongest training test:
    Verify encoder can fit a tiny target.
    """
    torch.manual_seed(0)

    encoder = make_encoder()

    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=5e-3,
    )

    xs = torch.randn(2, 16, INPUT_DIM).to(device)
    lengths = torch.tensor([16, 16]).to(device)

    target = torch.randn(2, 16, 64).to(device)

    losses = []

    for _ in range(200):
        optimizer.zero_grad()

        ys, _, _ = encoder(xs, lengths)

        loss = torch.nn.functional.mse_loss(
            ys,
            target,
        )

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    initial_loss = losses[0]
    final_loss = losses[-1]

    assert final_loss < initial_loss * 0.2, (
        "Tiny overfit test failed.\n"
        f"Initial loss: {initial_loss:.6f}\n"
        f"Final loss:   {final_loss:.6f}\n"
        "The encoder could not learn a tiny dataset."
    )


def test_state_dict_roundtrip():
    """
    Verify serialization correctness.
    """
    encoder1 = make_encoder()

    xs, lengths = make_inputs()

    y1, l1, _ = encoder1(xs, lengths)

    state = copy.deepcopy(
        encoder1.state_dict()
    )

    encoder2 = make_encoder()
    encoder2.load_state_dict(state)

    y2, l2, _ = encoder2(xs, lengths)

    assert torch.allclose(
        y1,
        y2,
        atol=1e-6,
        rtol=1e-6,
    ), (
        "Outputs differ after loading state_dict."
    )

    assert torch.equal(l1, l2), (
        "Output lengths differ after loading state_dict."
    )


def test_batch_independence():
    """
    Ensure samples do not interfere across batch dimension.
    """
    encoder = make_encoder()

    encoder.eval()

    x = torch.randn(1, TIME, INPUT_DIM).to(device)
    l = torch.tensor([TIME]).to(device)

    y_single, _, _ = encoder(x, l)

    xb = x.repeat(2, 1, 1)
    lb = l.repeat(2)

    y_batch, _, _ = encoder(xb, lb)

    assert torch.allclose(
        y_single,
        y_batch[0:1],
        atol=1e-5,
        rtol=1e-5,
    ), (
        "Output changed when identical sample was "
        "placed in a batch."
    )


@pytest.mark.parametrize(
    "seq_len",
    [1, 2, 4, 8, 16, 64],
)
def test_variable_sequence_lengths(seq_len):
    """
    Verify encoder handles very short and longer sequences.
    """
    encoder = make_encoder()

    xs = torch.randn(
        2,
        seq_len,
        INPUT_DIM,
    ).to(device)

    lengths = torch.tensor(
        [seq_len, seq_len],
        dtype=torch.long,
    ).to(device)

    ys, out_lengths, _ = encoder(
        xs,
        lengths,
    )

    assert ys.shape[1] == seq_len, (
        f"Sequence length changed unexpectedly. "
        f"Expected {seq_len}, got {ys.shape[1]}"
    )

    assert torch.equal(
        lengths,
        out_lengths,
    ), (
        f"Length propagation failed for "
        f"sequence length {seq_len}"
    )