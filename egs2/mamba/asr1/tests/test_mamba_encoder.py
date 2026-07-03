import torch
import torch.nn as nn
import torch.optim as optim

from espnet2.asr.encoder.mamba_encoder import MambaEncoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)


INPUT_DIM = 32
OUTPUT_DIM = 32
SEQ_LEN = 20

TRAIN_SAMPLES = 128

EPOCHS = 500

LR = 1e-2
TARGET_MSE = 1e-3


def generate_dataset():

    # W = torch.randn(INPUT_DIM, OUTPUT_DIM)

    # b = torch.randn(OUTPUT_DIM)

    x = torch.randn(TRAIN_SAMPLES, SEQ_LEN, INPUT_DIM)

    x_1 = torch.roll(x,1,dims=1)
    x_1[:,0,:] = 0

    x_2 = torch.roll(x,2,dims=1)
    x_2[:,0,:] = 0
    x_2[:,1,:] = 0
    

    y = (
        0.6 * x +
        0.3 * x_1 +
        0.1 * x_2
    )

    # y = x @ W + b

    return x, y


def build_model():

    # Minimal causal Mamba baseline for this synthetic sequence regression task.
    model = MambaEncoder(
        input_size=INPUT_DIM,
        output_size=OUTPUT_DIM,
        mamba_type="mamba1",
        hidden_size=64,
        num_blocks=1,
        d_state=8,
        d_conv=2,
        expand=1,
        conv_kernel_size=3,
        conv_num_layers=1,
        linear_units=32,
        dropout_rate=0.00,
    )

    return model.to(DEVICE)


def train():

    x, y = generate_dataset()

    x = x.to(DEVICE)
    y = y.to(DEVICE)

    lengths = torch.full(
        (TRAIN_SAMPLES,),
        SEQ_LEN,
        dtype=torch.long,
        device=DEVICE,
    )

    model = build_model()

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
    )

    losses = []

    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad()

        pred, _, _ = model(x, lengths)

        loss = criterion(pred, y)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

        if loss.item() < TARGET_MSE:
            print(
                f"Epoch {epoch:03d} | "
                f"Train MSE = {loss.item():.6e} | "
                f"early stop"
            )
            break

        # test_loss = 0

        if epoch % 20 == 0:

            # model.eval()

            # state = model.init_streaming_state()

            # y_pred, _ , state = model(x, ilens=lengths, prev_states=state)

            # loss_infer = criterion(
            #     y_pred,
            #     y,
            # )

            # test_loss += loss_infer.item()

            # test_loss /= len(test_loader)

            print(
                f"Epoch {epoch:03d} | "
                f"Train MSE = {loss.item():.6e} | "
                #f"Test MSE = {test_loss:.8e}"
            )


    print()

    print("Initial loss :", losses[0])
    print("Final loss   :", losses[-1])

    assert losses[-1] < losses[0], (
        "Training did not decrease the loss."
    )

    assert losses[-1] < 1e-3, (
        f"Encoder failed to overfit "
        f"(final MSE={losses[-1]:.4e})"
    )

    print("\nPASS: Encoder successfully overfit the dataset.")


if __name__ == "__main__":
    train()