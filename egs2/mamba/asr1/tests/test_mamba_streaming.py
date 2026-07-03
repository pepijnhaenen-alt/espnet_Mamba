import torch
import copy

from espnet2.asr.encoder.mamba_encoder import MambaEncoder
from espnet2.asr.state_spaces.s6 import Mamba1, Mamba2
#For Mamba2 simply replace Mamba1 with Mamba2.
from mamba_ssm.modules.mamba_simple import Mamba


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)


BATCH = 16
TIME = 40
DMODEL = 64


def main():

    # model_full = Mamba(
    #     d_model=DMODEL,
    #     d_state=16,
    #     d_conv=4,
    #     expand=1,
    # ).to(DEVICE)

    model_full = Mamba1(
        d_model=DMODEL,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout_rate=0.0,
        layer_idx=0,
    ).to(DEVICE)

    # model_full = MambaEncoder(
    #     input_size=DMODEL,
    #     output_size=DMODEL,
    #     num_blocks=2,
    #     dropout_rate=0.0,
    # ).to(DEVICE)

    model_full.eval()

    model_stream = copy.deepcopy(model_full)

    x = torch.randn(
        BATCH,
        TIME,
        DMODEL,
        device=DEVICE,
    )

    # ----------------------------------------------------
    # Full-sequence inference
    # ----------------------------------------------------

    state_full = model_full.default_state(
        BATCH,
        device=DEVICE,
    )

    with torch.no_grad():
        y_full, _ = model_full(x,state=state_full)

    # ----------------------------------------------------
    # Streaming inference
    # ----------------------------------------------------

    state_stream = model_stream.default_state(
        BATCH,
        device=DEVICE,
    )

    outputs = []

    with torch.no_grad():
        
        for t in range(TIME):

            x_t = x[:, t:t+1]

            y_t, state_stream = model_stream(
                x_t,
                state=state_stream,
            )

            # print(y_full[0][t], y_t, state_stream.key_value_memory_dict)
            # print(state_stream.seqlen_offset)

            outputs.append(y_t)

    y_stream = torch.cat(outputs, dim=1)

    # ----------------------------------------------------
    # Compare
    # ----------------------------------------------------

    abs_diff = (y_full - y_stream).abs()

    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    print(f"Maximum absolute difference : {max_diff:.8e}")
    print(f"Mean absolute difference    : {mean_diff:.8e}")

    atol = 1e-5
    rtol = 1e-4

    if torch.allclose(
        y_full,
        y_stream,
        atol=atol,
        rtol=rtol,
    ):
        print("\nPASS: Streaming output matches full-sequence output.")
    else:
        print("\nFAIL: Streaming output differs from full-sequence output.")

        idx = torch.argmax(abs_diff)
        idx = torch.unravel_index(idx, abs_diff.shape)

        print("\nLargest error at:")
        print(f"batch   = {idx[0]}")
        print(f"time    = {idx[1]}")
        print(f"channel = {idx[2]}")

        print(
            f"full     = {y_full[idx].item():.8f}"
        )
        print(
            f"stream   = {y_stream[idx].item():.8f}"
        )

        raise AssertionError(
            "Streaming and full inference are not equivalent."
        )


if __name__ == "__main__":
    main()