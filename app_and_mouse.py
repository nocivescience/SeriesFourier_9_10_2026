import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 40          # armónicos: k = -N..N  (2N+1 círculos)
T = 400         # puntos de la curva
STEPS = 3000    # pasos de entrenamiento


# ---------- 1. Dibujar con el mouse ----------
def dibujar_con_mouse():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Dibuja con el mouse (clic sostenido). Suelta para terminar.")
    (linea,) = ax.plot([], [], "k-")
    pts, estado = [], {"dibujando": False}

    def on_press(e):
        if e.inaxes == ax:
            estado["dibujando"] = True
            pts.clear()

    def on_move(e):
        if estado["dibujando"] and e.inaxes == ax:
            pts.append((e.xdata, e.ydata))
            linea.set_data(*zip(*pts))
            fig.canvas.draw_idle()

    def on_release(e):
        if estado["dibujando"]:
            estado["dibujando"] = False
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)
    plt.show()
    return np.array(pts)


def preparar_curva(pts):
    """Cierra la curva, la remuestrea a T puntos equiespaciados y la normaliza."""
    pts = np.vstack([pts, pts[0]])
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    u = np.linspace(0, d[-1], T, endpoint=False)
    z = np.interp(u, d, pts[:, 0]) + 1j * np.interp(u, d, pts[:, 1])
    z -= z.mean()
    z /= np.abs(z).max()
    return z


# ---------- 2. Modelo: coeficientes complejos c_k entrenables ----------
class SerieFourier(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("k", torch.arange(-n, n + 1, dtype=torch.float32))
        self.re = nn.Parameter(torch.randn(2 * n + 1) * 0.01)
        self.im = nn.Parameter(torch.randn(2 * n + 1) * 0.01)

    def coef(self):
        return torch.complex(self.re, self.im)

    def forward(self, t):
        fase = torch.outer(self.k, t)                      # (2N+1, T)
        e = torch.complex(torch.cos(fase), torch.sin(fase))
        return (self.coef()[:, None] * e).sum(0)           # z(t) = sum c_k e^{ikt}


def entrenar(z_obj):
    t = torch.linspace(0, 2 * math.pi, T + 1)[:-1]
    objetivo = torch.tensor(z_obj, dtype=torch.complex64)
    modelo = SerieFourier(N)
    opt = torch.optim.Adam(modelo.parameters(), lr=0.02)
    for paso in range(STEPS):
        opt.zero_grad()
        dif = modelo(t) - objetivo
        loss = (dif.real ** 2 + dif.imag ** 2).mean()
        loss.backward()
        opt.step()
        if paso % 300 == 0:
            print(f"paso {paso:4d}  loss = {loss.item():.6f}")
    return modelo


# ---------- 3. Animación con círculos y vectores ----------
def animar(modelo):
    c = modelo.coef().detach().numpy()
    k = modelo.k.numpy()
    orden = np.argsort(-np.abs(c))      # círculos grandes primero
    c, k = c[orden], k[orden]
    radios = np.abs(c)

    tf = np.linspace(0, 2 * np.pi, T, endpoint=False)
    terminos = c[:, None] * np.exp(1j * k[:, None] * tf[None, :])  # (K, T)
    fin = np.cumsum(terminos, axis=0)       # punta de cada vector
    inicio = fin - terminos                 # centro de cada círculo

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")

    th = np.linspace(0, 2 * np.pi, 60)
    circulos = [ax.plot([], [], color="steelblue", lw=0.6, alpha=0.4)[0] for _ in c]
    (vectores,) = ax.plot([], [], color="black", lw=1)
    (rastro,) = ax.plot([], [], color="crimson", lw=2)

    def update(f):
        for j, circ in enumerate(circulos):
            circ.set_data(inicio[j, f].real + radios[j] * np.cos(th),
                          inicio[j, f].imag + radios[j] * np.sin(th))
        puntos = np.r_[0, fin[:, f]]
        vectores.set_data(puntos.real, puntos.imag)
        rastro.set_data(fin[-1, : f + 1].real, fin[-1, : f + 1].imag)
        return [*circulos, vectores, rastro]

    anim = FuncAnimation(fig, update, frames=T, interval=20, blit=True, repeat=True)
    plt.show()
    return anim


if __name__ == "__main__":
    pts = dibujar_con_mouse()
    if len(pts) < 10:
        raise SystemExit("Dibujo demasiado corto, intenta de nuevo.")
    z = preparar_curva(pts)
    modelo = entrenar(z)
    animar(modelo)