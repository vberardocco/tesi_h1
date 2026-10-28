"""
Animazione di un trial reach-and-grasp di FALCON H1.

Ricostruisce visivamente il movimento a 7 DoF descritto dalla cinematica di riferimento
(`OpenLoopKinematics`), sincronizzato con l'attività neurale della popolazione e con le
fasi del task. Serve a *vedere* cosa il partecipante stava tentando di fare in ogni
momento del trial — cioè a dare un'intuizione concreta di ciò che i notebook 01 e 02
analizzano in forma numerica.

Cosa mostra il video
--------------------
  - **Pannello 3D**: la posizione dell'end-effector (`tx, ty, tz`), con la traiettoria
    completa in grigio e una scia colorata secondo la fase corrente del task.
  - **Mano schematica**: rotazione del polso (`rx`) e flessione dei tre gradi di presa
    (`g1` pollice, `g2` indice, `g3` anulare/mignolo).
  - **Attività neurale**: firing rate di popolazione (media sulle unità) con cursore
    temporale, così si vede la modulazione neurale mentre il movimento avviene.
  - **Velocità 7 DoF**: le tracce del target ufficiale del benchmark, con cursore.

Uso
---
    # dalla root del progetto, con l'ambiente `tesi-manifolds` attivo
    python -m tesi_h1.animate --trial 0
    python -m tesi_h1.animate --trial 3 --format gif --out figures/grasp_trial3.gif
    python -m tesi_h1.animate --list-trials          # elenca i trial disponibili

Note
----
  - La cinematica è in unità arbitrarie (a.u.): tutte le grandezze vengono normalizzate
    per-DoF sui percentili 1–99 della sessione, quindi la mano schematica è una lettura
    *relativa* del movimento, non una ricostruzione anatomica in scala.
  - Se `ffmpeg` non è disponibile lo script ripiega automaticamente su GIF.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from . import io as h1io


# --------------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------------

def _norm(x: np.ndarray, lo_hi: "tuple[float, float] | None" = None) -> np.ndarray:
    """Normalizza in [0, 1] usando i percentili 1–99 (robusto agli outlier).

    Se il DoF è costante nel trial restituisce 0.5 (posizione neutra), non 0: evita
    di disegnare una mano artificiosamente aperta o un polso ruotato al fondo corsa.
    """
    lo, hi = lo_hi if lo_hi is not None else (np.percentile(x, 1), np.percentile(x, 99))
    if hi - lo < 1e-12:
        return np.full_like(np.asarray(x, dtype=float), 0.5)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _base_tag(tag: str) -> str:
    """'Presentation3' -> 'Presentation'; gli altri tag restano invariati."""
    return tag.rstrip("234567") if tag.startswith("Presentation") else tag


def _rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# --------------------------------------------------------------------------------------
# Mano schematica
# --------------------------------------------------------------------------------------

# Geometria di base: mano vista **di lato**, palmo verticale e dita che escono a destra.
# Ogni dito è una catena di 2 segmenti; la flessione li arrotola verso il palmo.
# `sign` = verso della flessione: dita verso il basso (-1), pollice in opposizione (+1).
# L'ordine delle chiavi è g1, g2, g3 ed è lo stesso usato per colori, barre e legenda.
_FINGERS = {
    #  nome       base (x, y)     riposo            segmenti      sweep (prox, dist)      verso
    "g1": dict(base=(0.02, -0.40), rest=np.deg2rad(-58), seg=(0.34, 0.24),
               sweep=(88, 62), sign=+1, label="pollice"),
    "g2": dict(base=(0.24, 0.30),  rest=np.deg2rad(6),   seg=(0.46, 0.33),
               sweep=(100, 82), sign=-1, label="indice"),
    "g3": dict(base=(0.24, 0.02),  rest=np.deg2rad(-6),  seg=(0.43, 0.31),
               sweep=(100, 82), sign=-1, label="anulare/mignolo"),
}
_GRIP_COLORS = {"g1": "#d62728", "g2": "#1f77b4", "g3": "#2ca02c"}
_PALM = np.array([[-0.46, -0.42], [0.24, -0.42], [0.24, 0.46], [-0.46, 0.46]])
_WRIST = np.array([[-0.86, -0.28], [-0.46, -0.28], [-0.46, 0.32], [-0.86, 0.32]])


def _finger_points(cfg: dict, flex: float, roll: float) -> np.ndarray:
    """Punti (3, 2) della catena di un dito, data la flessione in [0, 1] e il roll del polso.

    flex = 0 -> dito esteso; flex = 1 -> dito completamente arrotolato sul palmo.
    Il giunto distale segue il prossimale, come nella flessione reale delle dita.
    """
    s = cfg["sign"]
    a1 = cfg["rest"] + s * flex * np.deg2rad(cfg["sweep"][0])   # giunto prossimale
    a2 = a1 + s * flex * np.deg2rad(cfg["sweep"][1])            # giunto distale
    p0 = np.array(cfg["base"], dtype=float)
    p1 = p0 + cfg["seg"][0] * np.array([np.cos(a1), np.sin(a1)])
    p2 = p1 + cfg["seg"][1] * np.array([np.cos(a2), np.sin(a2)])
    return np.stack([p0, p1, p2]) @ _rot2(roll).T


# --------------------------------------------------------------------------------------
# Costruzione dell'animazione (lavora su array puri: testabile senza NWB)
# --------------------------------------------------------------------------------------

def build_animation(
    kin: np.ndarray,
    vel: np.ndarray,
    pop_rate: np.ndarray,
    phase_lab: np.ndarray,
    bin_s: float = 0.02,
    fps: int = 25,
    decim: int = 2,
    title: str = "",
    trail_s: float = 2.0,
):
    """Costruisce la FuncAnimation di un trial.

    Parameters
    ----------
    kin : (T, 7)   posizione dei 7 DoF   [tx, ty, tz, rx, g1, g2, g3]
    vel : (T, 7)   velocità dei 7 DoF (target ufficiale FALCON)
    pop_rate : (T,) firing rate medio sulla popolazione, in Hz
    phase_lab : (T,) etichetta di fase per ogni bin
    decim : usa 1 frame ogni `decim` bin (a 20 ms, decim=2 -> 25 fps in tempo reale)
    """
    T = len(kin)
    t = np.arange(T) * bin_s
    frames = np.arange(0, T, decim)
    trail_n = max(2, int(trail_s / bin_s))

    # --- normalizzazioni ---------------------------------------------------------------
    xyz = kin[:, 0:3]
    roll = _norm(kin[:, 3]) * np.deg2rad(70) - np.deg2rad(35)      # ±35° di roll del polso
    grips = {g: _norm(kin[:, 4 + i]) for i, g in enumerate(("g1", "g2", "g3"))}
    base = np.array([_base_tag(p) if p else "" for p in phase_lab])

    def color_at(i: int) -> str:
        return h1io.PHASE_COLORS.get(base[i], "0.6")

    # --- figura ------------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(3, 3, height_ratios=[2.3, 1.0, 1.15], hspace=0.42, wspace=0.28,
                  left=0.06, right=0.985, top=0.90, bottom=0.08)
    ax3d = fig.add_subplot(gs[0, 0:2], projection="3d")
    axh = fig.add_subplot(gs[0, 2])
    axn = fig.add_subplot(gs[1, :])
    axv = fig.add_subplot(gs[2, :], sharex=axn)

    # --- pannello 3D: end-effector -----------------------------------------------------
    ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="0.85", lw=1.0, zorder=1)
    trail, = ax3d.plot([], [], [], lw=2.6, color="C0", zorder=2)
    head, = ax3d.plot([], [], [], "o", ms=11, mec="k", mew=0.8, color="C0", zorder=3)
    pad = max(0.06 * float(np.ptp(xyz, axis=0).max()), 1e-3)   # guardia: traiettoria degenere
    ax3d.set(xlabel="tx", ylabel="ty", zlabel="tz",
             xlim=(xyz[:, 0].min() - pad, xyz[:, 0].max() + pad),
             ylim=(xyz[:, 1].min() - pad, xyz[:, 1].max() + pad),
             zlim=(xyz[:, 2].min() - pad, xyz[:, 2].max() + pad),
             title="End-effector — traslazione 3D (tx, ty, tz)")
    ax3d.view_init(elev=22, azim=-58)
    ax3d.grid(True)

    # --- pannello mano -----------------------------------------------------------------
    wrist = plt.Polygon(_WRIST, closed=True, fc="0.80", ec="0.40", lw=1.2, zorder=1)
    palm = plt.Polygon(_PALM, closed=True, fc="0.90", ec="0.35", lw=1.5, zorder=2)
    axh.add_patch(wrist)
    axh.add_patch(palm)
    fing_lines, fing_joints = {}, {}
    for g, cfg in _FINGERS.items():
        ln, = axh.plot([], [], lw=7.5, solid_capstyle="round", solid_joinstyle="round",
                       color=_GRIP_COLORS[g], zorder=3)
        jt, = axh.plot([], [], "o", ms=4.5, color="w", mec=_GRIP_COLORS[g], mew=1.2, zorder=4)
        fing_lines[g], fing_joints[g] = ln, jt
    axh.set(xlim=(-1.25, 1.25), ylim=(-1.55, 1.30), aspect="equal",
            title="Configurazione della mano\nrx = roll del polso · g1–g3 = gradi di presa")
    axh.set_xticks([]); axh.set_yticks([]); axh.grid(False)
    for sp in axh.spines.values():
        sp.set_visible(False)

    # barre di flessione: una "traccia" grigia fissa + la barra colorata che cresce
    bar_y = {"g1": -1.12, "g2": -1.26, "g3": -1.40}
    bar_x0, bar_w = -0.72, 1.55
    bars, bar_txt = {}, {}
    for g, y in bar_y.items():
        axh.barh(y, bar_w, height=0.085, left=bar_x0, color="0.90", zorder=2)
        bars[g] = axh.barh(y, 0.0, height=0.085, left=bar_x0,
                           color=_GRIP_COLORS[g], zorder=3)[0]
        axh.text(bar_x0 - 0.06, y, g, ha="right", va="center", fontsize=8.5,
                 color=_GRIP_COLORS[g], fontweight="bold")
        bar_txt[g] = axh.text(bar_x0 + bar_w + 0.06, y, "", ha="left", va="center",
                              fontsize=7.5, color="0.35")
    roll_txt = axh.text(0.0, -0.90, "", ha="center", va="center", fontsize=8.5, color="0.35")

    # --- pannello neurale --------------------------------------------------------------
    axn.plot(t, pop_rate, color="C3", lw=1.3)
    axn.set(ylabel="Hz", title="Attività neurale — firing rate di popolazione (media sulle unità)",
            xlim=(t[0], t[-1]))
    cur_n = axn.axvline(t[0], color="k", lw=1.4, zorder=5)

    # --- pannello velocità -------------------------------------------------------------
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for j, lab in enumerate(h1io.KIN_LABELS):
        axv.plot(t, vel[:, j], lw=1.1, color=cols[j], label=lab)
    axv.set(xlabel="tempo nel trial (s)", ylabel="velocità (a.u./bin)",
            title="Cinematica 7 DoF — velocità (target ufficiale FALCON)")
    axv.legend(ncol=7, fontsize=8, loc="upper right")
    cur_v = axv.axvline(t[0], color="k", lw=1.4, zorder=5)

    # bande di fase sullo sfondo dei due pannelli temporali
    seen = []
    edges = np.flatnonzero(np.r_[True, base[1:] != base[:-1]])
    for k, a in enumerate(edges):
        b = edges[k + 1] if k + 1 < len(edges) else T
        tag = base[a]
        if tag in ("", "Presentation", "Intertrial"):
            continue
        for ax in (axn, axv):
            ax.axvspan(t[a], t[b - 1] + bin_s, color=h1io.PHASE_COLORS.get(tag, "0.8"),
                       alpha=0.16, lw=0, zorder=0)
        if tag not in seen:
            seen.append(tag)
    axn.legend(handles=[Patch(color=h1io.PHASE_COLORS.get(s, "0.8"), alpha=0.45, label=s)
                        for s in seen],
               ncol=min(len(seen), 8), fontsize=7.5, loc="upper center",
               bbox_to_anchor=(0.5, -0.20), frameon=False)

    suptitle = fig.suptitle("", fontsize=13, y=0.965)

    # --- update ------------------------------------------------------------------------
    def update(i: int):
        c = color_at(i)
        a = max(0, i - trail_n)
        trail.set_data(xyz[a:i + 1, 0], xyz[a:i + 1, 1])
        trail.set_3d_properties(xyz[a:i + 1, 2])
        trail.set_color(c)
        head.set_data([xyz[i, 0]], [xyz[i, 1]])
        head.set_3d_properties([xyz[i, 2]])
        head.set_color(c)

        r = roll[i]
        for g, cfg in _FINGERS.items():
            pts = _finger_points(cfg, grips[g][i], r)
            fing_lines[g].set_data(pts[:, 0], pts[:, 1])
            fing_joints[g].set_data([pts[1, 0]], [pts[1, 1]])
            bars[g].set_width(bar_w * grips[g][i])
            bar_txt[g].set_text(f"{grips[g][i]:.2f}")
        palm.set_xy(_PALM @ _rot2(r).T)
        wrist.set_xy(_WRIST @ _rot2(r).T)
        roll_txt.set_text(f"roll (rx) = {np.rad2deg(r):+.0f}°")

        cur_n.set_xdata([t[i], t[i]])
        cur_v.set_xdata([t[i], t[i]])
        ph = base[i] if base[i] else "—"
        suptitle.set_text(f"{title}    |    t = {t[i]:5.2f} s    |    fase: {ph}")
        suptitle.set_color(h1io.PHASE_COLORS.get(ph, "k") if ph in h1io.PHASE_COLORS else "k")
        return (trail, head, wrist, palm, cur_n, cur_v, suptitle, roll_txt,
                *fing_lines.values(), *fing_joints.values(),
                *bars.values(), *bar_txt.values())

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    return fig, anim


def save_animation(anim, out: Path, fps: int = 25, fmt: str = "auto") -> Path:
    """Salva in mp4 (ffmpeg) se disponibile, altrimenti in gif (pillow)."""
    out = Path(out)
    want_mp4 = fmt == "mp4" or (fmt == "auto" and out.suffix == ".mp4")
    if want_mp4:
        try:
            anim.save(str(out.with_suffix(".mp4")), writer=FFMpegWriter(fps=fps, bitrate=2400))
            return out.with_suffix(".mp4")
        except Exception as e:                       # ffmpeg assente o non funzionante
            print(f"[!] ffmpeg non disponibile ({type(e).__name__}), ripiego su GIF.")
    anim.save(str(out.with_suffix(".gif")), writer=PillowWriter(fps=fps))
    return out.with_suffix(".gif")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

DEFAULT_SESSION = ("data/raw/sub-HumanPitt-held-in-calib/"
                   "sub-HumanPitt-held-in-calib_ses-19250108T110520.nwb")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Animazione di un trial reach-and-grasp (FALCON H1).")
    ap.add_argument("--session", default=DEFAULT_SESSION, help="percorso del file NWB")
    ap.add_argument("--trial", type=int, default=0, help="indice del trial da animare")
    ap.add_argument("--out", default=None, help="file di output (.mp4 o .gif)")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--decim", type=int, default=2, help="1 frame ogni N bin (2 = tempo reale)")
    ap.add_argument("--sigma-ms", type=float, default=50.0, help="smoothing del firing rate")
    ap.add_argument("--format", choices=["auto", "mp4", "gif"], default="auto")
    ap.add_argument("--list-trials", action="store_true", help="elenca i trial e esce")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    sess_path = Path(args.session)
    if not sess_path.is_absolute():
        sess_path = root / sess_path
    if not sess_path.exists():
        raise SystemExit(f"File non trovato: {sess_path}\n"
                         f"Serve il dandiset in data/raw (symlink incluso nel repo).")

    matplotlib.use("Agg")
    s = h1io.load_session(sess_path)
    slices = h1io.trial_slices(s)

    if args.list_trials:
        for k, (a, b) in enumerate(slices):
            print(f"trial {k:>2}: bin {a:>6}–{b:<6}  durata {(b - a) * h1io.BIN_S:5.1f}s")
        return 0

    if not 0 <= args.trial < len(slices):
        raise SystemExit(f"--trial deve essere tra 0 e {len(slices) - 1}")

    a, b = slices[args.trial]
    rates = h1io.firing_rates(s, sigma_ms=args.sigma_ms)
    phase = h1io.phase_per_bin(s)

    out = Path(args.out) if args.out else root / "figures" / f"grasp_trial{args.trial}.mp4"
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, anim = build_animation(
        kin=s.kin[a:b], vel=s.vel[a:b], pop_rate=rates[a:b].mean(1),
        phase_lab=phase[a:b], bin_s=h1io.BIN_S, fps=args.fps, decim=args.decim,
        title=f"FALCON H1 · {s.session_id} · trial {args.trial}",
    )
    print(f"Rendering {len(range(0, b - a, args.decim))} frame…")
    written = save_animation(anim, out, fps=args.fps, fmt=args.format)
    plt.close(fig)
    print(f"✓ salvato in {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
