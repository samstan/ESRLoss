import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Data (example)
# -----------------------
torch.manual_seed(0)

# One covariate x, shape (n, 1)
x = torch.randn(100, 1)

# True relationship: y = 2x + 1 + noise
y = 2.0 * x + 1.0 + 0.1 * torch.randn(100, 1)

def esr(y_pred, y, alpha = 0.5, k = 5):
    return alpha * torch.dot(F.sigmoid(k * torch.flatten(y_pred)), torch.flatten(y)) + (1 - alpha) * F.mse_loss(y_pred, y)

# -----------------------
# Model definition
# -----------------------
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # One weight (slope) and one bias (intercept)
        self.linear = nn.Linear(1, 1, bias=True)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

# -----------------------
# Loss and optimizer
# -----------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# -----------------------
# Training loop
# -----------------------
num_epochs = 200

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(x)
    loss = esr(y_pred, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# -----------------------
# Learned parameters
# -----------------------
w = model.linear.weight.item()
b = model.linear.bias.item()

print(f"\nLearned weight (w): {w:.3f}")
print(f"Learned intercept (b): {b:.3f}")