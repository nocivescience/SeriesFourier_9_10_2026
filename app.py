import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------- 1. Curva objetivo (un corazón; cámbiala por la tuya) ----------
T = 500
t = torch.linspace(0, 2 * math.pi, T)
x_target = 16 * torch.sin(t) ** 3
y_target = 13 * torch.cos(t) - 5 * torch.cos(2 * t) - 2 * torch.cos(3 * t) - torch.cos(4 * t)
target = torch.stack([x_target, y_target]) / 16  # forma (2, T), normalizada


# ---------- 2. Modelo: serie de Fourier con coeficientes entrenables ----------
class FourierCurve(nn.Module):
    def __init__(self, n_harmonics=10):
        super().__init__()
        self.n = n_harmonics
        # fila 0 -> coeficientes de x(t), fila 1 -> coeficientes de y(t)
        self.a = nn.Parameter(torch.randn(2, n_harmonics + 1) * 0.01)  # cosenos
        self.b = nn.Parameter(torch.randn(2, n_harmonics + 1) * 0.01)  # senos

    def forward(self, t):
        k = torch.arange(0, self.n + 1, dtype=t.dtype)
        kt = k[:, None] * t[None, :]                 # (N+1, T)
        return self.a @ torch.cos(kt) + self.b @ torch.sin(kt)  # (2, T)


model = FourierCurve(n_harmonics=10)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

# ---------- 3. Entrenamiento ----------
for step in range(2000):
    opt.zero_grad()
    loss = ((model(t) - target) ** 2).mean()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        print(f"paso {step:4d}  loss = {loss.item():.6f}")

# ---------- 4. Dibujar ----------
with torch.no_grad():
    fit = model(t)

plt.figure(figsize=(5, 5))
plt.plot(*target, "--", color="gray", label="objetivo")
plt.plot(*fit, color="crimson", label=f"Fourier ({model.n} armónicos)")
plt.axis("equal")
plt.legend()
plt.show()