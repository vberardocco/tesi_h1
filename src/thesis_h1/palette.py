"""Colors and markers of the methods, identical in all figures produced by the thesis scripts.

Rules:
- every method has a fixed color and marker, assigned in the fixed order of the reference categorical palette
  (never cycled). With pairs adjacent in the order the palette passes the color-blind and normal-vision checks;
  with all pairs it does not (no order of 8 hues achieves this: FA and ridge_channels are hard to tell apart even with
  normal vision, CEBRA and FA for protanopes). Hence, in every figure with more than 3 methods the identity is also
  given by the marker and by labels or a legend, never by color alone;
- the variants of a method (CEBRA-Behavior, GPFA + Poisson regression, LFADS with the controller, ...) take the
  color of the family (FAMILY) and are distinguished by fill or hatching, not by a new hue;
- references that are not a method (chance, clock, smoothing + GLM, ridge_rates) are in neutral gray;
- text and axes are in neutral ink, never in the color of a series.
In cross-session figures A and D the color indicates the condition (COND_STYLE) within panels or groups already split
by method: this must be stated in the caption. The figures saved by notebooks 02-04 use the matplotlib colors and are
not regenerated (the notebooks are not re-run); the comparison between methods for the thesis is in scripts/fig_latent_structure.py.
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
    """The base method of a name (e.g. "CEBRA-Behavior" -> "CEBRA"); the name itself if it is not a known variant."""
    return FAMILY.get(name, name)


def color(name: str) -> str:
    """Color of the method (or of its family); neutral gray for anything that is not a method."""
    return METHOD_COLOR.get(family(name), NEUTRAL)


def marker(name: str) -> str:
    return METHOD_MARKER.get(family(name), "o")


def style_axes(ax):
    """Light grid, axes and ticks in secondary ink, without the top and right spines."""

    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
