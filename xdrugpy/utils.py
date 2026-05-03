import builtins
import itertools
import sys
import shlex
import inspect
import scipy.cluster.hierarchy as sch
from functools import wraps
from textwrap import dedent
from typing import get_args, Union, Any, get_origin, get_type_hints
from types import UnionType
from collections import namedtuple, defaultdict
from shutil import rmtree
from tempfile import mkdtemp
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib import pyplot as plt, axes
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage
from collections import namedtuple
from strenum import StrEnum
from enum import Enum
from pymol.parser import __file__ as _parser_filename

from pymol import Qt, cmd as pm, parsing


@pm.extend
def plot(filename: str | None = None):
    """Show or save the current Matplotlib plot."""
    from matplotlib import pyplot as plt
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def configure_matplotlib(style='default', params=None):
    """Configure Matplotlib for use in XDrugPy."""
    import matplotlib.style
    import matplotlib.colors
    from matplotlib import pyplot as plt
    from cycler import cycler
    
    matplotlib.use("Qt5Agg")
    matplotlib.style.use(style)
    plt.rcParams.update({
        **{
            'font.size': 14,
            'figure.figsize': (10, 6),
            'svg.fonttype': 'none',
            'axes.prop_cycle': cycler(color=reversed(matplotlib.colors.XKCD_COLORS))
        },
        **(params or {}),
    })


class AligMethod(StrEnum):
    ALIGN = "align"
    SUPER = "super"
    CEALIGN = "cealign"
    FIT = "fit"
    

class Selection(str):
    pass


@pm.new_command
def align_groups(
    mobile_groups: Selection,
    target: Selection,
    align_method: AligMethod = AligMethod.CEALIGN,
):
    for mobile in mobile_groups.split():
        pm.extra_fit(
            f"{mobile}.protein",
            f"{target}.protein",
            method=align_method
        )
        for inner in pm.get_object_list(f"{mobile}.*"):
            if ".protein" in inner:
                continue
            pm.matrix_copy(f"{mobile}.protein", inner)


def plot_hca_base(
    dists,
    labels,
    linkage_method,
    color_threshold,
    only_medoids,
    annotate,
    axis=None,
    vmin=None,
    vmax=None,
    enable_heatmap=False,
    rename_leafs=None,
    no_plot=False
):
    if isinstance(axis, axes.Axes):
        fig = axis.get_figure()
        fig.clear()
    elif not no_plot:
        fig, ax = plt.subplots(constrained_layout=True)
        ax.remove()
        
    ax_dend_top = None
    if enable_heatmap:
        gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 1], wspace=0.01, hspace=0.01)
        ax_dend_top = fig.add_subplot(gs[0])
        ax_heat = fig.add_subplot(gs[1])
    elif not no_plot:
        gs = fig.add_gridspec(1, 1, height_ratios=[1], wspace=0.01, hspace=0.01)
        ax_dend_top = fig.add_subplot(gs[0])
    
    for leaf_node, new_label in (rename_leafs or {}).items():
        idx = labels.index(leaf_node)
        labels[idx] = new_label
    
    Z = linkage(dists, method=linkage_method, optimal_ordering=True)
    dendro = sch.dendrogram(
        Z,
        labels=labels,
        orientation="top",
        color_threshold=color_threshold,
        distance_sort=True,
        leaf_rotation=90,
        ax=ax_dend_top,
        no_labels=enable_heatmap,
        no_plot=no_plot,
    )
    if not no_plot and color_threshold is not None and color_threshold > 0:
        ax_dend_top.axhline(color_threshold, color="gray", ls="--")
        ax_dend_top.set_ylim(bottom=-0.005)

    dists = distance.squareform(dists)
    X = dists
    X = X[dendro["leaves"], :]
    X = X[:, dendro["leaves"]]

    if enable_heatmap:
        ax_heat.set_xticks(range(len(dendro["ivl"])), dendro["ivl"])
        ax_heat.set_yticks(range(len(dendro["ivl"])), dendro["ivl"])
        ax_heat.tick_params(axis="x", rotation=90)
        ax_heat.yaxis.tick_right()
        image = ax_heat.imshow(X, aspect="auto", vmin=vmin, vmax=vmax)
        
        if not annotate:
            fig.colorbar(image, ax=ax_heat, shrink=0.8)
        if annotate:
            xmin = vmin or X.min()
            xmax = vmax or X.max()
            
            for i1, x1 in enumerate(X):
                for i2, x2 in enumerate(X):
                    y = X[i1, i2]
                    if (y - xmin)/(xmax - xmin) >= 0.5:
                        color = "black"
                    else:
                        color = "white"
                    label = f"{y:.2f}"
                    ax_heat.text(i2, i1, label, color=color, ha="center", va="center")

    # Calcular a soma das distâncias para cada ponto do cluster
    cl_d_sums = defaultdict(float)
    colors = set(dendro["leaves_color_list"]) - {"C0"}

    for color in colors:
        for leaf1_idx, leaf_label1, color1 in zip(
            dendro["leaves"], dendro["ivl"], dendro["leaves_color_list"]
        ):
            if color != color1:
                continue

            # Soma das distâncias de leaf1 para todos os outros pontos do mesmo cluster
            d_sum = 0
            for leaf2_idx, leaf_label2, color2 in zip(
                dendro["leaves"], dendro["ivl"], dendro["leaves_color_list"]
            ):
                if color != color2:
                    continue
                # Note: X contém distâncias, então valores menores = mais próximos
                d_sum += dists[leaf1_idx, leaf2_idx]
            
            cl_d_sums[(color, leaf_label1)] = d_sum

    # Encontrar o medoid (ponto com MENOR soma de distâncias) para cada cluster
    medoids = {}
    for color in colors:
        min_d = float("inf")
        min_leaf = None
        
        for (color1, leaf_label1), d_sum in cl_d_sums.items():
            if color1 != color:
                continue
            
            # Medoid é o ponto com a MENOR soma de distâncias
            if d_sum < min_d:
                min_d = d_sum
                min_leaf = leaf_label1
        
        assert min_leaf is not None
        for (color1, leaf_label1), d_sum in cl_d_sums.items():
            if d_sum == min_d:
                if color not in medoids:
                    medoids[color] = set()
                medoids[color].add(leaf_label1)
    if not no_plot:
        medoids_labels = set(itertools.chain.from_iterable(medoids.values()))
        label_to_color = dict(zip(dendro["ivl"], dendro["leaves_color_list"]))
        if enable_heatmap:
            ticklabels = [*ax_heat.get_xticklabels(), *ax_heat.get_yticklabels()]
        else:
            ticklabels = ax_dend_top.get_xticklabels()

        for label in ticklabels:
            color = label_to_color[label.get_text()]
            label.set_color(color)
            if color in medoids and label.get_text() in medoids[color]:
                label.set_fontstyle("italic")
                label.set_fontweight('bold')
        
            if only_medoids and color_threshold > 0.0:
                if label.get_text() not in medoids_labels:
                    label.set_visible(False)
    return dendro, medoids
