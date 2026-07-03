import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from espnet2.asr.state_spaces.s6 import Mamba1
from espnet2.asr.state_spaces.s6 import Mamba2

from mamba_ssm.modules.mamba_simple import Mamba
# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)

INPUT_DIM = 32
OUTPUT_DIM = 32 #16
SEQ_LEN = 200

N_TRAIN = 1024
N_TEST = 256

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3

# ----------------------------------------------------------
# Ground-truth linear mapping
# ----------------------------------------------------------

W_true = torch.randn(INPUT_DIM, OUTPUT_DIM)
b_true = torch.randn(OUTPUT_DIM)

def generate_dataset(n_samples):

    x = torch.randn(
        n_samples,
        SEQ_LEN,
        INPUT_DIM,
    )

    y = torch.zeros(n_samples, SEQ_LEN, OUTPUT_DIM,)
    
    for t in range(2, SEQ_LEN):
        y[:, t] = (
            0.8 * y[:,t-1] + 0.2* x[:,t] 
        )
    # y = x @ W_true + b_true

    lengths = torch.full(
        (n_samples,),
        SEQ_LEN,
        dtype=torch.long,
    )

    return x, y, lengths

x_train, y_train, len_train = generate_dataset(
N_TRAIN
)

x_test, y_test, len_test = generate_dataset(
N_TEST
)

train_loader = DataLoader(
TensorDataset(
x_train,
y_train,
len_train,
),
batch_size=BATCH_SIZE,
shuffle=True,
)

test_loader = DataLoader(
TensorDataset(
x_test,
y_test,
len_test,
),
batch_size=BATCH_SIZE,
)

# ----------------------------------------------------------
# Model
# ----------------------------------------------------------

class MambaLinearProbe(nn.Module):

    def __init__(self):

        super().__init__()

        # self.mamba = nn.Linear(INPUT_DIM, INPUT_DIM)

        # self.mamba = lambda x: (x, "")

        # self.mamba = Mamba1(
        #     d_model=INPUT_DIM,
        #     d_state=16,
        #     d_conv=2,
        #     expand=1,
        #     dropout_rate=0.0,
        #     layer_idx = 0,
        # )

        self.mamba = Mamba(
            d_model=INPUT_DIM,
            d_state=INPUT_DIM,
            d_conv=2,
            expand=1,
        ).to("cuda")

        self.output_proj = nn.Linear(
            INPUT_DIM,
            OUTPUT_DIM,
        )

    def forward(self, x, state=None):
        if state is None:
            y = self.mamba(x)
            # y, _ = self.mamba(x)
                
            # y = self.output_proj(y)
            # y = self.output_proj(x)

            return y, None
        
        y, state = self.mamba(x, inference_params=state)
        # y = self.output_proj(y)
        return y, state

    def default_state(self):
        # return self.mamba.default_state()
        return None

model = MambaLinearProbe().to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
model.parameters(),
lr=LR,
)

# ----------------------------------------------------------
# Sanity check
# ----------------------------------------------------------

x = torch.randn(
2,
SEQ_LEN,
INPUT_DIM,
device=DEVICE,
)

state = model.default_state()

with torch.no_grad():

    y, _ = model(x, state)

    print("Output shape:", y.shape)

    assert y.shape == (2, SEQ_LEN, OUTPUT_DIM,)

# ----------------------------------------------------------
# Training
# ----------------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for x, y_true, _ in train_loader:

        x = x.to(DEVICE)
        y_true = y_true.to(DEVICE)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(
            y_pred,
            y_true,
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # for n,p in model.named_parameters():
    #     if p.grad is not None:
    #         print(n, p.grad.norm())

    model.eval()

    test_loss = 0.0

    with torch.no_grad():

        for x, y_true, _ in test_loader:

            x = x.to(DEVICE)
            y_true = y_true.to(DEVICE)

            state = model.default_state()

            y_pred, state = model(x, state)

            loss = criterion(
                y_pred,
                y_true,
            )

            test_loss += loss.item()

    test_loss /= len(test_loader)

    print(
        f"Epoch {epoch:03d} | "
        f"Train MSE = {train_loss:.8e} | "
        f"Test MSE = {test_loss:.8e}"
    )
# ----------------------------------------------------------
# Final diagnostics
# ----------------------------------------------------------

model.eval()

with torch.no_grad():

    x = x_test.to(DEVICE)

    y_true = y_test.to(DEVICE)

    state = model.default_state()

    y_pred, state = model(x, state)

    mse = torch.mean(
        (y_pred - y_true) ** 2
    ).item()

    max_err = torch.max(
        torch.abs(y_pred - y_true)
    ).item()

    print()
    print("====================================")
    print(f"FINAL MSE : {mse:.10e}")
    print(f"MAX ABS ERROR : {max_err:.10e}")
    print("====================================")

# ----------------------------------------------------------
# Parameter gradient inspection
# ----------------------------------------------------------

print()
print("Gradient statistics:")

for name, param in model.named_parameters():

    if param.grad is not None:

        print(
            f"{name:50s} "
            f"{param.grad.norm().item():.6e}"
        )