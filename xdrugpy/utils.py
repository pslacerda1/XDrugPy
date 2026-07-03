import itertools
import os
import subprocess
import signal
import sys
from pathlib import Path
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
from pymol.exporting import _resn_to_aa as RESN_TO_AA
from pymol import Qt, cmd as pm
from pymol_new_command import new_command


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


@new_command
def count_molecules(sel: Selection, quiet: bool = True) -> int:
    """
    Returns the number of distinct molecules in a given selection.
    """

    sel_copy = "__selcopy"
    pm.select(sel_copy, sel)
    num_objs = 0
    atoms_in_sel = pm.count_atoms(sel_copy)
    while atoms_in_sel > 0:
        num_objs += 1
        pm.select(sel_copy, "%s and not (bm. first %s)" % (sel_copy, sel_copy))
        atoms_in_sel = pm.count_atoms(sel_copy)
    if not quiet:
        print(f"Number of molecules: {num_objs}")
    return num_objs


def plot_hca_base(
    dists,
    labels,
    linkage_method,
    only_medoids,
    annotate,
    vmin=None,
    vmax=None,
    rename_leafs=None,
    nclusters=-1,
    color_threshold=-1.0,
    dendrogram_axis=None,
    heatmap_axis=None,
):
    for leaf_node, new_label in (rename_leafs or {}).items():
        idx = labels.index(leaf_node)
        labels[idx] = new_label
    if nclusters != -1.0 and color_threshold != -1.0:
        raise ValueError("Cannot specify both nclusters and color_threshold.")
    Z = linkage(dists, method=linkage_method)
    if nclusters != -1.0:
        # Calculate the distance threshold that corresponds to the desired number of clusters
        color_threshold = threshold_for_k_clusters(Z, nclusters)

    if dendrogram_axis:
        if isinstance(dendrogram_axis, (str, Path, bool)):
            _, dendro_ax = plt.subplots()

        elif isinstance(dendrogram_axis, axes.Axes):
            dendro_ax = dendrogram_axis
    else:
        dendro_ax = None

    dendro = sch.dendrogram(
        Z,
        labels=labels,
        color_threshold=color_threshold,
        distance_sort=True,
        leaf_rotation=90,
        ax=dendro_ax,
        no_plot=not dendro_ax
    )
    if dendro_ax and color_threshold > 0:
        dendro_ax.axhline(color_threshold, color="gray", ls="--")
        dendro_ax.set_ylim(bottom=-0.005)

    dists = distance.squareform(dists)
    X = dists
    X = X[dendro["leaves"], :]
    X = X[:, dendro["leaves"]]

    if heatmap_axis:
        if isinstance(heatmap_axis, (str, Path)):
            _, heat_ax = plt.subplots()

        elif isinstance(heatmap_axis, axes.Axes):
            heat_ax = heatmap_axis
    else:
        heat_ax = None
    
    if heat_ax:
        heat_ax.set_xticks(range(len(dendro["ivl"])), dendro["ivl"])
        heat_ax.set_yticks(range(len(dendro["ivl"])), dendro["ivl"])
        heat_ax.tick_params(axis="x", rotation=90)
        heat_ax.yaxis.tick_right()
        image = heat_ax.imshow(X, aspect="auto", vmin=vmin, vmax=vmax)
        
        if not annotate:
            heat_ax.get_figure().colorbar(image, ax=heat_ax, shrink=0.8)
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
                    ticklabel = f"{y:.2f}"
                    heat_ax.text(i2, i1, ticklabel, color=color, ha="center", va="center")

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



    ticklabels = []
    if dendro_ax:
        ticklabels = [
            *ticklabels,
            *dendro_ax.get_xticklabels()
        ]
    if heat_ax:
        ticklabels = [
            *ticklabels,
            *heat_ax.get_xticklabels(),
            *heat_ax.get_yticklabels()
        ]

    medoids_labels = set(itertools.chain.from_iterable(medoids.values()))
    label_to_color = dict(zip(dendro["ivl"], dendro["leaves_color_list"]))
    for ticklabel in ticklabels:
        color = label_to_color[ticklabel.get_text()]
        ticklabel.set_color(color)
        if color in medoids and ticklabel.get_text() in medoids[color]:
            ticklabel.set_fontstyle("italic")
            ticklabel.set_fontweight('bold')
    
        if only_medoids and color_threshold > 0.0:
            if ticklabel.get_text() not in medoids_labels:
                ticklabel.set_visible(False)

    if dendrogram_axis:
        fig = dendro_ax.get_figure(True)
        fig.set_layout_engine('compressed')
        if isinstance(dendrogram_axis, (str, Path)):
            fig.savefig(dendrogram_axis)
        else:
            fig.show()
    
    if heatmap_axis:
        fig = heat_ax.get_figure(True)
        fig.set_layout_engine('compressed')
        if isinstance(heatmap_axis, (str, Path)):
            fig.savefig(str(heatmap_axis))
        else:
            fig.show()
    
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
        "clustalo -i - --outfmt=clu --wrap=45",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True,
    )
    output_fasta, err = proc.communicate(input_fasta)
    if err:
        raise Exception(f"Clustal Omega error: {err}")
    
    # joining multiline sequences
    output = output_fasta.split('\n')[3:]
    sequences = {}
    while output:
        line = output.pop(0)
        if line.strip() == "":
            continue
        if line[0] != ' ':
            name, seq = line.split(maxsplit=1)
        else:
            name = 'CLUSTALO'
            seq = line[-len(seq):]
        sequences[name] = sequences.get(name, "") + seq

    # skiping gaps and keeping counters ok
    clu = sequences.pop('CLUSTALO')
    omega = {}
    for (title, seq), sele in zip(sequences.items(), seles):
        atoms = [a for a in pm.get_model(f"({sele}) & present & guide & polymer").atom]
        at_ix = 0
        for aln_ix, (seq_char, clu_char) in enumerate(zip(seq, clu)):
            if seq_char == '-':
                continue
            at = atoms[at_ix]
            if clu_char in conservation:
                if title not in omega:
                    omega[title] = []
                omega[title].append(Residue(
                    at.model, at.index, at.resi, at.chain, at.resn, seq_char, clu_char
                ))
            at_ix += 1
        
    assert at_ix == len(atoms), (
        f"Alignment/atom mismatch for {title}: consumed {at_ix} atoms, "
        f"but selection has {len(atoms)} guide atoms"
    )
    omega = {
        replaced_dict[title]: omega[title]
        for title in sorted(omega, key=replaced_list.index)
    }
    return omega



from pymol import Qt

QWidget = Qt.QtWidgets.QWidget
QFileDialog = Qt.QtWidgets.QFileDialog
QFormLayout = Qt.QtWidgets.QFormLayout
QPushButton = Qt.QtWidgets.QPushButton
QSpinBox = Qt.QtWidgets.QSpinBox
QDoubleSpinBox = Qt.QtWidgets.QDoubleSpinBox
QLineEdit = Qt.QtWidgets.QLineEdit
QCheckBox = Qt.QtWidgets.QCheckBox
QVBoxLayout = Qt.QtWidgets.QVBoxLayout
QHBoxLayout = Qt.QtWidgets.QHBoxLayout
QDialog = Qt.QtWidgets.QDialog
QComboBox = Qt.QtWidgets.QComboBox
QTabWidget = Qt.QtWidgets.QTabWidget
QLabel = Qt.QtWidgets.QLabel
QTableWidget = Qt.QtWidgets.QTableWidget
QTableWidgetItem = Qt.QtWidgets.QTableWidgetItem
QGroupBox = Qt.QtWidgets.QGroupBox
QHeaderView = Qt.QtWidgets.QHeaderView
QTextEdit = Qt.QtWidgets.QTextEdit

QtCore = Qt.QtCore
QIcon = Qt.QtGui.QIcon
QTextCursor = Qt.QtGui.QTextCursor



def kill_process(proc):
    """
    Mata o processo corretamente, incluindo subprocessos
    """
    if proc.poll() is not None:
        return  # Já terminou
    
    try:
        if sys.platform == 'win32':
            # Windows - usar taskkill para matar árvore
            subprocess.call(
                ['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            # Unix - matar grupo de processos
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # Se não terminou, força
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
    
    except ProcessLookupError:
        pass  # Processo já morreu
    except Exception as e:
        print(f"Error killing process: {e}")
        # Último recurso
        try:
            proc.kill()
            proc.wait()
        except:
            pass

class PyMOLComboObjectBox(QComboBox):

    def __init__(self):
        super().__init__()
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.setEditText("")

    def showPopup(self):
        currentText = self.currentText().strip()
        selections = pm.get_names("selections", enabled_only=False)
        objects = pm.get_names("objects", enabled_only=False)
        self.clear()
        self.addItems("(%s)" % s for s in selections)
        self.addItems(objects)
        if currentText != "":
            self.setCurrentText(currentText)
        super().showPopup()