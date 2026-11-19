"""Paper-style figures for experiments in the build pod. The build agent writes this file into
the scratch directory before the experiment starts; importing it applies the style.

    import fractal_plot as fp
    fig, ax = fp.figure("single")
    ax.plot(seeds, scores, label="Expert persona")
    ax.set(xlabel="Seed", ylabel="Agreement rate (%)")
    fp.save(fig, "agreement_by_seed")

Name a condition by `fp.label(key)`: its display name from the plan, never a shorthand of the
results key ("pos. control"). `save` writes `figures/<name>.png` at print size and 300 dpi,
refuses any other format, and refuses a figure whose text holds a results key or shorthand.
"""

import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.text import Text  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
DISPLAY_NAMES_FILE = Path(__file__).resolve().parent / "display_names.json"
"""The plan's display names by results key, written into the pod before the run."""
INSIDE_WORDS = re.compile(
    r"(?<![\w.])[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+(?![\w])"
    r"|(?<![\w])(?:pos|neg|ctrl|ctl|cond|ctx|cfg|w/o|w/)(?:\.|(?![\w]))",
    re.IGNORECASE,
)
"""A results key (snake_case) or a shorthand of one, which a figure must not show."""
MATH = re.compile(r"\$[^$]*\$")
"""Mathtext such as $x_i$, whose subscripts are not results keys."""
DPI = 300
MAX_BYTES = 1_500_000
WIDTHS_INCHES = {"single": 3.5, "double": 7.0}
DEFAULT_ASPECT = 0.72

OKABE_ITO = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000")
"""Colorblind-safe colors with at least 3:1 contrast on white; pair them with MARKERS."""
MARKERS = ("o", "s", "^", "D", "v")
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"

PAPER_STYLE = {
    "figure.facecolor": "white",
    "figure.dpi": 100,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.titlelocation": "left",
    "axes.labelsize": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "axes.prop_cycle": cycler(color=OKABE_ITO) + cycler(marker=MARKERS),
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "lines.markeredgewidth": 0.5,
    "errorbar.capsize": 2,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "image.cmap": SEQUENTIAL_CMAP,
    "savefig.dpi": DPI,
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "savefig.bbox": "standard",
}


def use_paper_style() -> None:
    plt.rcParams.update(PAPER_STYLE)


def figure(width: str = "single", height: float | None = None, nrows: int = 1, ncols: int = 1):
    """A figure at print width ("single" 3.5 in or "double" 7 in) with constrained layout,
    so labels are not clipped. Returns (fig, axes) like `plt.subplots`."""
    inches = WIDTHS_INCHES[width]
    size = (inches, height or inches * DEFAULT_ASPECT * nrows / ncols)
    return plt.subplots(nrows, ncols, figsize=size, layout="constrained")


def display_names() -> dict[str, str]:
    if not DISPLAY_NAMES_FILE.exists():
        return {}
    return json.loads(DISPLAY_NAMES_FILE.read_text())


def label(key: str, width: int | None = None) -> str:
    """The plan's display name for a results key, wrapped at `width` characters if given."""
    names = display_names()
    if key not in names:
        known = ", ".join(sorted(names)) or "none"
        raise KeyError(f"no display name for {key!r}; the plan names: {known}")
    return textwrap.fill(names[key], width) if width else names[key]


def inside_words(fig) -> list[str]:
    """Every results key or shorthand in the figure's text: ticks, legend, labels, titles."""
    texts = (MATH.sub("", text.get_text()) for text in fig.findobj(Text))
    return sorted({match.group(0) for text in texts for match in INSIDE_WORDS.finditer(text)})


def save(fig, name: str) -> Path:
    """Write figures/<name>.png at print size and 300 dpi, close the figure, return the path."""
    path = FIGURES_DIR / name
    if path.suffix not in ("", ".png") or path.parent != FIGURES_DIR:
        raise ValueError(f"name must be a plain file name saved as PNG: {name!r}")
    found = inside_words(fig)
    if found:
        raise ValueError(
            f"the figure shows words from inside the run: {', '.join(found)}; name each "
            "condition with fp.label(key) and spell out the rest"
        )
    path = path.with_suffix(".png")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", dpi=DPI)
    plt.close(fig)
    size = path.stat().st_size
    if size > MAX_BYTES:
        path.unlink()
        raise ValueError(f"{path.name} is {size} bytes, above {MAX_BYTES}: simplify or shrink it")
    print(f"saved {path}")
    return path


use_paper_style()
