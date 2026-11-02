"""Colori e marcatori dei metodi, uguali in tutte le figure prodotte dagli script della tesi.

Regole:
- ogni metodo ha un colore e un marcatore fissi, assegnati nell'ordine fisso della palette categoriale di riferimento
  (mai ciclata). Con le coppie vicine nell'ordine la palette supera i controlli di daltonismo e di visione normale;
  con tutte le coppie no (nessun ordine di 8 tinte ci riesce: FA e ridge_channels si distinguono poco anche con visione
  normale, CEBRA e FA poco per i protanopi). Quindi in ogni figura con più di 3 metodi l'identità è data anche dal
  marcatore e da etichette o legenda, mai dal solo colore;
- le varianti di un metodo (CEBRA-Behavior, GPFA + regressione di Poisson, LFADS con il controller, ...) prendono il
  colore della famiglia (FAMILY) e si distinguono con il riempimento o il tratteggio, non con una tinta nuova;
- i riferimenti che non sono un metodo (caso, orologio, smoothing + GLM, ridge_rates) sono in grigio neutro;
- testo e assi sono in inchiostro neutro, mai nel colore di una serie.
Nelle figure A e D del cross-sessione il colore indica la condizione (COND_STYLE) dentro pannelli o gruppi già divisi
per metodo: va detto nella didascalia. Le figure salvate dai notebook 02-04 hanno i colori di matplotlib e non si
rigenerano (i notebook non si rieseguono); il confronto tra metodi per la tesi è in scripts/fig_latent_structure.py.
"""
from __future__ import annotations

CAT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
METHODS = ("PCA", "FA", "GPFA", "Isomap", "UMAP", "CEBRA", "LFADS", "ridge_channels")
METHOD_COLOR = dict(zip(METHODS, CAT))
METHOD_MARKER = dict(zip(METHODS, "osD^vPXh"))
METHOD_STYLE = {m: (METHOD_COLOR[m], METHOD_MARKER[m]) for m in METHODS}

FAMILY = {"CEBRA-Time": "CEBRA", "CEBRA-Behavior": "CEBRA", "GPFA alla pari": "GPFA",
          "GPFA + regressione di Poisson": "GPFA", "LFADS con il controller": "LFADS", "25 PC smussati": "PCA",
          "counts": "ridge_channels", "ridge_h50": "ridge_channels"}

COND_STYLE = {"zero_shot": (CAT[0], "o"), "recenter": (CAT[1], "s"), "align": (CAT[2], "^"), "fewshot": (CAT[3], "D"),
              "fewshot_gain": (CAT[5], "X"), "scratch": (CAT[4], "v"), "fa_native": (CAT[6], "P")}

INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#7a7975"


def family(name: str) -> str:
    """Il metodo di base di un nome (es. "CEBRA-Behavior" -> "CEBRA"); il nome stesso se non è una variante nota."""
    return FAMILY.get(name, name)


def color(name: str) -> str:
    """Colore del metodo (o della sua famiglia); grigio neutro per ciò che non è un metodo."""
    return METHOD_COLOR.get(family(name), NEUTRAL)


def marker(name: str) -> str:
    return METHOD_MARKER.get(family(name), "o")


def style_axes(ax):
    """Griglia leggera, assi e tacche in inchiostro secondario, senza i bordi in alto e a destra."""
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
