import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

T = 400            # puntos de la curva
K = 16             # armónicos de entrada (features de Fourier)
HIDDEN = 128       # neuronas por capa oculta
PASOS = 2000       # pasos de entrenamiento
N_CIRCULOS = 60    # círculos que se dibujan en la animación


# ---------- Red neuronal: t -> (x, y) ----------
class RedCurva(nn.Module):
    def __init__(self, k=K, h=HIDDEN):
        super().__init__()
        self.register_buffer("k", torch.arange(1, k + 1, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(2 * k, h), nn.Tanh(),   # capa oculta 1
            nn.Linear(h, h), nn.Tanh(),       # capa oculta 2
            nn.Linear(h, 2),                  # salida (x, y)
        )

    def forward(self, t):
        fase = t[:, None] * self.k[None, :]
        feats = torch.cat([torch.sin(fase), torch.cos(fase)], dim=1)  # (T, 2K)
        return self.net(feats)


def preparar(pts):
    """Cierra la curva, remuestrea a T puntos y devuelve complejo en unidades originales."""
    pts = np.vstack([pts, pts[0]])
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    u = np.linspace(0, d[-1], T, endpoint=False)
    return np.interp(u, d, pts[:, 0]) + 1j * np.interp(u, d, pts[:, 1])


# ---------- Interfaz ----------
fig, (ax, axl) = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [3, 2]})
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_aspect("equal")
ax.axis("off")
ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ls=":", color="gray"))
ax.set_title("Dibuja con el mouse (clic sostenido). Nuevo clic = borrar y empezar. Tecla 'c' = limpiar.")

axl.set_yscale("log")
axl.set_xlabel("paso")
axl.set_ylabel("pérdida (MSE)")
axl.set_title(f"Red: t → [sin/cos(kt)] ({2*K}) → {HIDDEN} → tanh → {HIDDEN} → tanh → 2", fontsize=9)

(linea_dibujo,) = ax.plot([], [], "k-", lw=2)
(linea_red,) = ax.plot([], [], color="crimson", lw=1.5)
(linea_loss,) = axl.plot([], [], color="steelblue")

estado = {"dibujando": False, "accion": None, "cancelar": False}
pts = []
artistas = []   # círculos y vectores de la animación


def limpiar_animacion():
    for a in artistas:
        a.remove()
    artistas.clear()


def limpiar_todo():
    estado["cancelar"] = True
    pts.clear()
    limpiar_animacion()
    linea_dibujo.set_data([], [])
    linea_red.set_data([], [])
    linea_loss.set_data([], [])
    fig.canvas.draw_idle()


def on_press(e):
    if e.inaxes == ax:
        limpiar_todo()                 # borra la curva anterior
        estado["dibujando"] = True


def on_move(e):
    if estado["dibujando"] and e.inaxes == ax:
        pts.append((e.xdata, e.ydata))
        linea_dibujo.set_data(*zip(*pts))
        fig.canvas.draw_idle()


def on_release(e):
    if estado["dibujando"]:
        estado["dibujando"] = False
        if len(pts) >= 10:
            estado["accion"] = "entrenar"


def on_key(e):
    if e.key == "c":
        limpiar_todo()


fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("key_press_event", on_key)


# ---------- Entrenar la red (en vivo) y animar epiciclos ----------
def entrenar_y_animar():
    estado["cancelar"] = False
    z = preparar(np.array(pts))
    centro = z.mean()
    escala = np.abs(z - centro).max()
    zn = (z - centro) / escala
    objetivo = torch.tensor(np.stack([zn.real, zn.imag], axis=1), dtype=torch.float32)

    t = torch.linspace(0, 2 * math.pi, T + 1)[:-1]
    modelo = RedCurva()
    opt = torch.optim.Adam(modelo.parameters(), lr=5e-3)
    perdidas = []

    for paso in range(PASOS):
        opt.zero_grad()
        loss = ((modelo(t) - objetivo) ** 2).mean()
        loss.backward()
        opt.step()
        perdidas.append(loss.item())

        if paso % 20 == 0:
            with torch.no_grad():
                out = modelo(t).numpy()
            linea_red.set_data(out[:, 0] * escala + centro.real, out[:, 1] * escala + centro.imag)
            linea_loss.set_data(range(len(perdidas)), perdidas)
            axl.relim()
            axl.autoscale_view()
            plt.pause(0.001)
            if estado["cancelar"]:
                return

    # --- Epiciclos a partir de la salida de la red (FFT) ---
    with torch.no_grad():
        out = modelo(t).numpy()
    zc = (out[:, 0] + 1j * out[:, 1]) * escala + centro
    c = np.fft.fft(zc) / T
    k = np.fft.fftfreq(T, 1 / T)
    idx = np.argsort(-np.abs(c))[:N_CIRCULOS]
    c, k = c[idx], k[idx]
    radios = np.abs(c)

    tf = np.linspace(0, 2 * np.pi, T, endpoint=False)
    terminos = c[:, None] * np.exp(1j * k[:, None] * tf[None, :])
    fin = np.cumsum(terminos, axis=0)
    inicio = fin - terminos

    linea_red.set_alpha(0.25)
    th = np.linspace(0, 2 * np.pi, 60)
    circulos = [ax.plot([], [], color="steelblue", lw=0.6, alpha=0.4)[0] for _ in c]
    (vectores,) = ax.plot([], [], color="black", lw=1)
    (rastro,) = ax.plot([], [], color="darkorange", lw=2)
    artistas.extend([*circulos, vectores, rastro])

    while not estado["cancelar"] and plt.fignum_exists(fig.number):
        for f in range(T):
            for j, circ in enumerate(circulos):
                circ.set_data(inicio[j, f].real + radios[j] * np.cos(th),
                              inicio[j, f].imag + radios[j] * np.sin(th))
            puntos = np.r_[0, fin[:, f]]
            vectores.set_data(puntos.real, puntos.imag)
            rastro.set_data(fin[-1, : f + 1].real, fin[-1, : f + 1].imag)
            plt.pause(0.015)
            if estado["cancelar"]:
                break


if __name__ == "__main__":
    plt.show(block=False)
    while plt.fignum_exists(fig.number):
        if estado["accion"] == "entrenar":
            estado["accion"] = None
            linea_red.set_alpha(1.0)
            entrenar_y_animar()
        plt.pause(0.05)