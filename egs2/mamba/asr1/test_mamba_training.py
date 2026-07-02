# train_linear_mapping.py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from espnet2.asr.encoder.mamba_encoder import MambaEncoder
from espnet2.asr.state_spaces.s6 import Mamba1
from espnet2.asr.state_spaces.s6 import Mamba2

from mamba_ssm.modules.mamba_simple import Mamba

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)

INPUT_DIM = 32
OUTPUT_DIM = 32 #16
SEQ_LEN = 20

N_TRAIN = 512
N_TEST = 128

BATCH_SIZE = 32
EPOCHS = 50


# ---------------------------------------------------------
# Generate synthetic linear mapping
# ---------------------------------------------------------

W_true = torch.eye(INPUT_DIM, OUTPUT_DIM) #torch.randn(INPUT_DIM, OUTPUT_DIM)
b_true = torch.zeros(1,OUTPUT_DIM) #torch.randn(OUTPUT_DIM)


def generate_dataset(n_samples):
    x = torch.randn(n_samples, SEQ_LEN, INPUT_DIM)

    y = x @ W_true + b_true

    lengths = torch.full(
        (n_samples,),
        SEQ_LEN,
        dtype=torch.long,
    )

    return x, y, lengths


x_train, y_train, len_train = generate_dataset(N_TRAIN)
x_test, y_test, len_test = generate_dataset(N_TEST)

train_loader = DataLoader(
    TensorDataset(x_train, y_train, len_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    TensorDataset(x_test, y_test, len_test),
    batch_size=BATCH_SIZE,
)

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = Mamba(
    d_model=INPUT_DIM,
    d_state=16,
    d_conv=4,
    expand=2,
).to("cuda")

# model = Mamba1(
#     d_model=INPUT_DIM,
#     d_state=INPUT_DIM,
#     conv_num_layers = 0,
#     dropout_rate = 0.0,
#     expand=1,
#     layer_idx = 0,
# ).to(DEVICE)

# model = MambaEncoder(
#     input_size=INPUT_DIM,
#     output_size=OUTPUT_DIM,
#     num_blocks=1,
#     conv_num_layers = 0,
#     dropout_rate=0.0,
# ).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)

# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for x, y, lengths in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        lengths = lengths.to(DEVICE)

        optimizer.zero_grad()

        pred = model(x)
        # pred, _ = model(x) 
        # pred, _, _ = model(x, lengths)

        loss = criterion(pred, y)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()

    test_loss = 0.0

    with torch.no_grad():

        for x, y, lengths in test_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            lengths = lengths.to(DEVICE)

            pred = model(x)
            #pred, _ = model(x) 
            # pred, _, _ = model(x, lengths)

            loss = criterion(pred, y)

            test_loss += loss.item()

    test_loss /= len(test_loader)

    print(
        f"Epoch {epoch:03d} | "
        f"Train MSE = {train_loss:.6e} | "
        f"Test MSE = {test_loss:.6e}"
    )

# ---------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------

model.eval()

with torch.no_grad():

    x = x_test[:8].to(DEVICE)
    lengths = len_test[:8].to(DEVICE)

    pred = model(x)
    # pred, _ = model(x) 
    # pred, _, _ = model(x, lengths)

    target = y_test[:8].to(DEVICE)

    mse = ((pred - target) ** 2).mean()

print()
print(f"FINAL TEST MSE = {mse.item():.8e}")



