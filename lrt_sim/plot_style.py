"""Shared GRL/AGU figure style for lrt_sim plotting scripts.

GRL plot & graphic specifications encoded here:
  * Width 50-170 mm, height <= 228 mm         -> :func:`figsize_mm` (warns + clamps)
  * Raster >= 300 dpi (600 dpi combination
    halftones, 1200 dpi line art)             -> DPI constants + :func:`save_grl`
  * Vector output preferred                    -> :func:`save_grl` writes PDF alongside PNG
  * Sans-serif (Helvetica/Arial), 8-10 pt at
    final printed size                         -> :func:`apply_grl_style`
  * Line weights >= 0.25 pt                    -> enforced by the rcParams defaults
  * Colorblind-friendly, perceptually uniform
    colors                                     -> Okabe-Ito cycle, cividis default cmap

Usage pattern for a script whose figures go into the manuscript/SI::

    from plot_style import apply_grl_style, figsize_mm, save_grl, add_panel_label
    apply_grl_style()                               # once, at entry
    fig, ax = plt.subplots(figsize=figsize_mm(FULL_WIDTH_MM, 100))
    add_panel_label(ax, '(a)')
    save_grl(fig, 'fig/SI/my_figure')               # -> .png (300 dpi) + .pdf

Prefer the rcParams defaults over per-call ``fontsize=``/``linewidth=`` so a
single style change propagates everywhere. Sizes here are *print* sizes: build
the figure at its true published width (170 mm full page / 95 mm single column)
so the 8-10 pt fonts are genuinely 8-10 pt after typesetting.
"""

import os
import warnings

import matplotlib as mpl
from cycler import cycler

MM_PER_INCH = 25.4

# GRL geometry limits (mm).
MIN_WIDTH_MM = 50.0
COLUMN_WIDTH_MM = 95.0    # single-column width
FULL_WIDTH_MM = 170.0     # full-page width
MAX_HEIGHT_MM = 228.0

# GRL raster resolutions (dpi).
DEFAULT_DPI = 300         # color / grayscale
HALFTONE_DPI = 600        # combination halftones (image + line overlays)
LINEART_DPI = 1200        # pure line art

# Okabe-Ito colorblind-safe categorical palette (fixed assignment order).
OKABE_ITO = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # bluish green
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#CC79A7',  # reddish purple
    '#F0E442',  # yellow
    '#000000',  # black
]

# Perceptually uniform defaults (GRL-recommended family).
SEQUENTIAL_CMAP = 'cividis'
SEQUENTIAL_CMAP_ALT = 'viridis'


def figsize_mm(width_mm=FULL_WIDTH_MM, height_mm=None, aspect=0.62):
    """Figure size in inches from print dimensions in mm, checked against GRL limits.

    ``aspect`` (height/width) is used when ``height_mm`` is omitted. Out-of-range
    dimensions are clamped to the GRL limits with a warning, so a figure can
    never silently violate the spec.
    """
    if height_mm is None:
        height_mm = width_mm * aspect
    if not (MIN_WIDTH_MM <= width_mm <= FULL_WIDTH_MM):
        warnings.warn(
            f'GRL width must be {MIN_WIDTH_MM:g}-{FULL_WIDTH_MM:g} mm; '
            f'clamping {width_mm:g} mm.', stacklevel=2)
        width_mm = min(max(width_mm, MIN_WIDTH_MM), FULL_WIDTH_MM)
    if height_mm > MAX_HEIGHT_MM:
        warnings.warn(
            f'GRL height must be <= {MAX_HEIGHT_MM:g} mm; clamping '
            f'{height_mm:g} mm.', stacklevel=2)
        height_mm = MAX_HEIGHT_MM
    return (width_mm / MM_PER_INCH, height_mm / MM_PER_INCH)


def apply_grl_style(base_fontsize=9.0):
    """Set matplotlib rcParams to the GRL specification.

    Fonts land in the 8-10 pt band around ``base_fontsize`` (ticks/legend one
    step below, titles one step above); every default line weight stays above
    the 0.25 pt floor; PNG saves default to 300 dpi; PDF/PS text stays editable
    (TrueType, fonttype 42) for vector submission.
    """
    small = base_fontsize - 1.0
    large = base_fontsize + 1.0
    mpl.rcParams.update({
        # --- fonts -----------------------------------------------------------
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'font.size': base_fontsize,
        'axes.labelsize': base_fontsize,
        'axes.titlesize': large,
        'figure.titlesize': large,
        'xtick.labelsize': small,
        'ytick.labelsize': small,
        'legend.fontsize': small,
        'legend.title_fontsize': base_fontsize,
        # --- line weights (all >= 0.25 pt) ------------------------------------
        'lines.linewidth': 1.2,
        'lines.markersize': 4.0,
        'axes.linewidth': 0.6,
        'grid.linewidth': 0.4,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'patch.linewidth': 0.6,
        'hatch.linewidth': 0.4,
        # --- color -----------------------------------------------------------
        'axes.prop_cycle': cycler(color=OKABE_ITO),
        'image.cmap': SEQUENTIAL_CMAP,
        # --- output ----------------------------------------------------------
        'figure.dpi': 150,               # screen preview only
        'savefig.dpi': DEFAULT_DPI,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,              # keep text as editable TrueType
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        # --- misc ------------------------------------------------------------
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.7',
        'axes.axisbelow': True,
    })


def add_panel_label(ax, label, x=0.0, y=1.01, fontsize=None):
    """Bold panel tag ('(a)', '(b)', ...) above the top-left panel boundary."""
    if fontsize is None:
        fontsize = mpl.rcParams['axes.titlesize']
    return ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
                   fontweight='bold', ha='left', va='bottom')


def save_grl(fig, path_base, dpi=DEFAULT_DPI, formats=('png', 'pdf'), **kwargs):
    """Save one figure in raster + vector form per the GRL resolution rules.

    ``path_base`` has no extension; one file per entry in ``formats`` is
    written next to it (PNG at ``dpi``; PDF/EPS/SVG are resolution-free).
    Extra kwargs go to ``fig.savefig``.
    """
    root, ext = os.path.splitext(path_base)
    if ext.lower() in ('.png', '.pdf', '.eps', '.svg', '.tif', '.tiff'):
        path_base = root
    written = []
    for fmt in formats:
        out = f'{path_base}.{fmt}'
        fig.savefig(out, dpi=dpi, bbox_inches='tight', **kwargs)
        written.append(out)
    return written
