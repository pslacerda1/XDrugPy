import itertools
import os
import subprocess
import scipy.cluster.hierarchy as sch
from collections import defaultdict
from shutil import rmtree
from matplotlib import pyplot as plt, axes
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, fcluster
from collections import namedtuple
from functools import lru_cache
from strenum import StrEnum
from pymol.parser import __file__ as _parser_filename

from pymol import Qt, cmd as pm


class Selection(str):
    pass

Residue = namedtuple("Residue", "model index resi chain resn oneletter conservation")



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
    

def run(command, log=True, cwd=None, env=os.environ):
    if log:
        print("RUNNING PROCESS:", command)
    ret = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        shell=True,
        env=env,
    )
    output = ret.stdout.decode(errors="replace")
    success = ret.returncode == 0
    return output, success




def get_color_threshold(Z, k):
    """
    Calculates the color_threshold for a dendrogram to visualize k clusters.
    
    Args:
        Z: The linkage matrix (from scipy.cluster.hierarchy.linkage)
        k: The desired number of clusters
        
    Returns:
        float: The threshold value to pass to the dendrogram function
    
    Author:
        Gemini
    """
    if k < 2:
        # If k=1, the threshold must be higher than the maximum distance
        return Z[-1, 2] + 1.0
    
    # The (k-1)th merge from the end creates the k-th cluster.
    # We take the average between the distance that creates k-1 clusters 
    # and the distance that creates k clusters.
    
    # Distance that results in k clusters
    dist_k = Z[-k, 2]
    
    # Distance that results in k-1 clusters
    dist_k_minus_1 = Z[-(k-1), 2]
    
    return (dist_k + dist_k_minus_1) / 2


def threshold_for_k_clusters(Z, k):
    """
    Retorna um color_threshold para obter k clusters.

    """
    if k <= 1:
        return float('inf')

    k = min(k, len(Z))
    return (Z[-(k - 1), 2] + Z[-k, 2]) / 2

def plot_hca_base(
    dists,
    labels,
    linkage_method,
    only_medoids,
    annotate,
    axis=None,
    vmin=None,
    vmax=None,
    enable_heatmap=False,
    rename_leafs=None,
    nclusters=-1,
    color_threshold=-1.0,
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
    if nclusters != -1.0 and color_threshold != -1.0:
        raise ValueError("Cannot specify both nclusters and color_threshold.")
    Z = linkage(dists, method=linkage_method)
    if nclusters != -1.0:
        # Calculate the distance threshold that corresponds to the desired number of clusters
        Z = linkage(dists, method=linkage_method)
        color_threshold = threshold_for_k_clusters(Z, nclusters)
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
    if not no_plot and color_threshold > 0:
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



def clustal_omega(seles, conservation, titles=None):
    replaced_dict = {}
    replaced_list = []
    if not titles:
        titles = seles
    input_fasta = ''
    for sele, title in zip(seles, titles):
        query = f'({sele}) and present and guide and polymer'

        title_replaced = title.replace(' ', '_')
        replaced_dict[title_replaced] = title
        replaced_list.append(title_replaced)

        input_fasta += (
            ">" + title_replaced + 
            pm
            .get_fastastr(query, key='model')
            .removeprefix(f'>{sele}')
        )
    
    proc = subprocess.Popen(
        "clustalo -i - --outfmt=clustal",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        shell=True,
        text=True,
    )
    output, err = proc.communicate(input_fasta)
    if err:
        raise Exception(f"Clustal Omega error: {err}")
    
    # joining multiline sequences
    output = output.split('\n')[3:]
    sequences = {}
    while output:
        line = output.pop(0)
        if line.strip() == "":
            continue
        if line[0] != ' ':
            name, seq = line.split()
        else:
            name = 'CLUSTALO'
            seq = line[-len(seq):]
        sequences[name] = sequences.get(name, "") + seq

    # skiping gaps and keeping counters ok
    clu = sequences.pop('CLUSTALO')
    len_aln = len(clu)
    omega = {}
    for (title, seq), sele in zip(sequences.items(), seles):
        local_ix = 0
        atoms = [a for a in pm.get_model(f"%{sele} & present & guide & polymer").atom]
        for aln_ix in range(len_aln):
            seq_char = seq[aln_ix]
            clu_char = clu[aln_ix]
            if seq_char != '-':
                if clu_char in conservation:
                    assert local_ix < len(atoms)
                    at = atoms[local_ix]
                    if title not in omega:
                        omega[title] = []
                    omega[title].append(Residue(
                        at.model, at.index, int(at.resi), at.chain, at.resn, seq_char, clu_char
                    ))
                local_ix += 1
    
    omega = {
        replaced_dict[title]: omega[title]
        for title in sorted(omega, key=replaced_list.index)
    }
    return omega
