import itertools
import subprocess
import os
import atexit
import sys
import io
import signal
import scipy.cluster.hierarchy as sch
from collections import namedtuple, defaultdict
from shutil import rmtree
from tempfile import mkdtemp
from pymol import Qt, cmd as pm, parsing
from contextlib import contextmanager
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib import pyplot as plt, axes
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage
from collections import namedtuple
from functools import lru_cache
from rich.console import Console
from rich.traceback import Traceback
from rich import terminal_theme
from strenum import StrEnum


QStandardPaths = Qt.QtCore.QStandardPaths


RESOURCES_DIR = Path(
    QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
)
RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

LIGAND_LIBRARIES_DIR = Path(RESOURCES_DIR / "libs/ligands/")
LIGAND_LIBRARIES_DIR.mkdir(parents=True, exist_ok=True)

RECEPTOR_LIBRARIES_DIR = Path(RESOURCES_DIR / "libs/receptors/")
RECEPTOR_LIBRARIES_DIR.mkdir(parents=True, exist_ok=True)

TEMPDIR = Path(mkdtemp(prefix="XDrugPy-"))


def clear_temp():
    rmtree(TEMPDIR)


atexit.register(clear_temp)


Residue = namedtuple("Residue", "model index resi chain name x y z oneletter")


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


class Selection(str):
    pass


def _bool_func(value: str):
    if isinstance(value, str):
        if value.lower() in ["yes", "1", "true", "on"]:
            return True
        elif value.lower() in ["no", "0", "false", "off"]:
            return False
        else:
            raise Exception("Invalid boolean value: %s" % value)
    elif isinstance(value, bool):
        return value
    else:
        raise Exception(f"Unsuported boolean flag {value}")


def declare_command(name, function=None, _self=pm):
    if function is None:
        name, function = name.__name__, name

    if function.__code__.co_argcount != len(function.__annotations__):
        raise Exception("All command options must be annotated.")
    from functools import wraps
    import inspect
    from pathlib import Path
    from enum import Enum
    import traceback

    spec = inspect.getfullargspec(function)

    kwargs_ = {}
    args_ = spec.args[:]

    defaults = list(spec.defaults or [])

    args2_ = args_[:]
    while args_ and defaults:
        kwargs_[args_.pop(-1)] = defaults.pop(-1)

    funcs = {}
    for idx, (var, func) in enumerate(spec.annotations.items()):
        funcs[var] = func

    @wraps(function)
    def inner(*args, **kwargs):
        frame = traceback.format_stack()[-2]
        caller = frame.split('"', maxsplit=2)[1]
        if caller.endswith("pymol/parser.py"):
            kwargs = {**kwargs_, **kwargs, **dict(zip(args2_, args))}
            kwargs.pop("_self", None)
            for arg in kwargs.copy():
                if funcs[arg] is _bool_func or issubclass(funcs[arg], bool):
                    funcs[arg] = _bool_func
                kwargs[arg] = funcs[arg](kwargs[arg])
            return function(**kwargs)
        else:
            try:
                return function(*args, **kwargs)
            except Exception:
                display_exception()
                raise

    name = function.__name__
    _self.keyword[name] = [inner, 0, 0, ",", parsing.STRICT]
    _self.kwhash.append(name)
    _self.help_sc.append(name)
    return inner


@contextmanager
def mpl_axis(ax, **kwargs):
    if not ax:
        _, new_ax = plt.subplots(**kwargs)
    elif isinstance(ax, (str, Path)):
        output_file = ax
        _, new_ax = plt.subplots(**kwargs)
    elif isinstance(ax, Axes):
        new_ax = ax
    yield new_ax
    if not ax:
        plt.show()
    elif isinstance(ax, (str, Path)):
        plt.savefig(output_file)


@declare_command
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


def plot_hca_base(dists, labels, linkage_method, color_threshold, hide_threshold, annotate, axis):
    if isinstance(axis, axes.Axes):
        fig = axis.get_figure()
        fig.clear()
    else:
        fig, ax = plt.subplots(constrained_layout=True)
        ax.remove()
    
    gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 1], wspace=0.01, hspace=0.01)
    ax_dend_top = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    
    Z = linkage(dists, method=linkage_method, optimal_ordering=True)
    dendro = sch.dendrogram(
        Z,
        labels=labels,
        orientation="top",
        color_threshold=color_threshold,
        distance_sort=True,
        leaf_rotation=90,
        ax=ax_dend_top,
        no_labels=True,
    )
    if color_threshold > 0:
        ax_dend_top.axhline(color_threshold, color="gray", ls="--")
    ax_dend_top.set_ylim(bottom=-0.005)

    dists = distance.squareform(dists)

    X = dists
    X = X[dendro["leaves"], :]
    X = X[:, dendro["leaves"]]

    ax_heat.set_xticks(range(len(dendro["ivl"])), dendro["ivl"])
    ax_heat.set_yticks(range(len(dendro["ivl"])), dendro["ivl"])
    ax_heat.tick_params(axis="x", rotation=90)
    ax_heat.yaxis.tick_right()
    ax_heat.imshow(1-X, aspect="auto", vmin=0, vmax=1)

    if annotate:
        for i1, x1 in enumerate(X):
            for i2, x2 in enumerate(X):
                y = 1-X[i1, i2]
                if y >= 0.5:
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

            if color == 'C3':
                pass
            
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

    ticklabels = [*ax_heat.get_xticklabels(), *ax_heat.get_yticklabels()]
    for color in colors:
        for label in ticklabels:
            if label.get_text() in medoids[color]:
                label.set_color(color)
                label.set_fontstyle("italic")
    
    if hide_threshold and color_threshold > 0.0:
        visible_labels = set(itertools.chain.from_iterable(medoids.values()))
        for label in ticklabels:
            if label.get_text() not in visible_labels:
                label.set_visible(False)
    
    if not axis:
        fig.show()
    elif isinstance(axis, (str, Path)):
        fig.savefig(axis)
    return dendro, medoids


@lru_cache(25000)
def get_residue_from_object(obj, idx):
    res = []
    pm.iterate_state(
        -1,
        f"%{obj} & index {idx}",
        "res.append(Residue(model, int(index), int(resi), chain, name, float(x), float(y), float(z), oneletter))",
        space={"res": res, "Residue": Residue},
    )
    return res[0]



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


def display_exception():
    """Display exception with Rich formatting on GUI."""
    
    if pm.gui.get_qtwindow():
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback = Traceback.from_exception(
            exc_type,
            exc_value, 
            exc_traceback,
            show_locals=True,
            width=120,
            extra_lines=3,
            word_wrap=True,
        )
        console = Console(
            record=True,
            file=io.StringIO(),  # null handler
            width=120,
            tab_size=4,
        )
        console.print(traceback)

    
        dialog = QDialog()
        dialog.setWindowTitle("Error Display")
        dialog.setGeometry(100, 100, 900, 700)
        dialog.setWindowModality(QtCore.Qt.ApplicationModal)

        layout = QVBoxLayout(dialog)
        
        text_edit = QTextEdit()
        layout.addWidget(text_edit)
        
        # Export to HTML with inline styles
        html = console.export_html(
            inline_styles=True,
            code_format="<pre>{code}</pre>",
            theme=terminal_theme.NIGHT_OWLISH
        )
        text_edit.setHtml(html)
        text_edit.setReadOnly(True)
        text_edit.moveCursor(QTextCursor.End)

        dialog.exec_()
    else:
        raise



def kill_process(proc):
    """
    Mata processo corretamente, incluindo subprocessos
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