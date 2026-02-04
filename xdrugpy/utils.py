import builtins
import itertools
import subprocess
import os
import atexit
import sys
import io
import signal
import shlex
import inspect
import scipy.cluster.hierarchy as sch
from contextlib import wraps
from textwrap import dedent
from typing import get_args, Union, Any, get_origin, get_type_hints
from types import UnionType
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
from strenum import StrEnum
from enum import Enum


from pymol.parser import __file__ as _parser_filename

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


Residue = namedtuple("Residue", "model index resi chain resn oneletter conservation")


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

        
class ArgumentParsingError(ValueError):
    "Error on argument parsing."

    def __init__(self, arg_name, message):
        message = dedent(message).strip()
        if arg_name:
            s = f"Failed at parsing '{arg_name}'. {message}"
        else:
            s = message
        super().__init__(s)


def _into_types(var, type, value):
    # Untyped string
    if type == Any:
        return value
            
    # Boolean flags
    elif type is bool:
        if isinstance(value, bool):
            return value
        trues = ["yes", "1", "true", "on", "y"]
        falses = ["no", "0", "false", "off", "n"]
        if value.lower() in trues:
            return True
        elif value.lower() in falses:
            return False
        else:
            raise ArgumentParsingError(
                var,
                f"Can't parse {value!r} as bool."
                f" Supported true values are {', '.join(trues)}."
                f" Supported false values are {', '.join(falses)}."
            )
    
    # Types from typing module
    elif origin := get_origin(type):

        if origin in {Union, UnionType}:
            funcs = get_args(type)
            for func in funcs:
                try:
                    return _into_types(None, func, value)
                except:
                    continue
            raise ArgumentParsingError(
                var,
                f"Can't parse {value!r} into {type}."
                f" The parser tried each union type and none was suitable."
            )
        
        elif issubclass(origin, tuple):
            funcs = get_args(type)
            if funcs:
                values = shlex.split(value)
                if len(funcs) > 0 and len(funcs) != len(values):
                    raise ArgumentParsingError(
                        var,
                        f"Can't parse {value!r} into {type}."
                        f" The number of tuple arguments are incorrect."
                    )
                try:
                    return tuple(_into_types(None, f, v) for f, v in zip(funcs, values))
                except:
                    raise ArgumentParsingError(
                        var,
                        f"Can't parse {value!r} into {type}."
                        f" One or more tuple values are of incorrect types."
                    )
            else:
                return tuple(shlex.split(value))

        elif issubclass(origin, list):
            funcs = get_args(type)
            if len(funcs) == 1:
                func = funcs[0]
                return [_into_types(None, func, a) for a in shlex.split(value)]
            return shlex.split(value)
    
    elif sys.version_info >= (3, 11) and issubclass(type, StrEnum):
        try:
            return type(value)
        except:
            names = [e.value for e in list(type)]
            raise ArgumentParsingError(
                var,
                f"Invalid value for {type.__name__}."
                f" Accepted values are {', '.join(names)}."
            )

    # Specific types must go before other generic types
    #   isinstance(type, builtins.type) comes after
    elif issubclass(type, Enum):
        value = type.__members__.get(value)
        if value is None:
            raise ArgumentParsingError(
                var,
                f"Invalid value for {type.__name__}."
                f" Accepted values are {', '.join(type.__members__)}."
            )
        return value
    
    # Generic types must accept str as single argument to __init__(s)
    elif isinstance(type, builtins.type):
        try:
            return type(value)
        except Exception as exc:
            raise ArgumentParsingError(
                var,
                f"Invalid value {value!r} for custom type {type.__name__}."
                f" The type must accept str as the solo argument to __init__(s)."
            ) from exc


def new_command(name, function=None, _self=pm):

    if function is None:
        name, function = name.__name__, name

    # docstring text, if present, should be dedented
    if function.__doc__ is not None:
        function.__doc__ = dedent(function.__doc__).strip()

    # Resolve strings into real class objects (PEP 563).
    try:
        resolved_hints = get_type_hints(
            function,
            globalns=sys.modules[function.__module__].__dict__
        )
    except Exception:
        resolved_hints = function.__annotations__

    # Analysing arguments
    sign = inspect.signature(function)
    
    # Inner function that will be callable every time the command is executed
    @wraps(function)
    def inner(*args, **kwargs):
        caller = sys._getframe(1).f_code.co_filename
        # It was called from command line or pml script, so parse arguments
        if caller == _parser_filename:
            # special _self argument
            kwargs.pop("_self", None)
            new_kwargs = {}
            for var, param in sign.parameters.items():
                if var in kwargs:
                    value = kwargs[var]
                    # special 'quiet' argument
                    if var == 'quiet' and isinstance(value, int):
                        new_kwargs[var] = bool(value)
                    else:
                        actual_type = resolved_hints.get(var, param.annotation)
                        new_kwargs[var] = _into_types(var, actual_type, value)
                else:
                    if param.default is sign.empty:
                        raise RuntimeError(f"Unknow variable '{var}'.")
            defaults = {
                k: v.default for k, v in sign.parameters.items()
                if v.default is not sign.empty
            }
            final_kwargs = {
                **defaults,
                **new_kwargs
            }
            return function(**final_kwargs)

        # It was called from Python, so pass the arguments as is
        else:
            return function(*args, **kwargs)
    
    _self.keyword[name] = [inner, 0, 0, ',', parsing.STRICT]
    _self.kwhash.append(name)
    _self.help_sc.append(name)
    
    # Accessor to the original function so bypass the stack extraction.
    # The purpose is optimization (loops, for instance).
    inner.func = inner.__wrapped__ 
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


@new_command
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


def plot_hca_base(dists, labels, linkage_method, color_threshold, hide_threshold, annotate, axis, vmin=None, vmax=None):
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
    ax_heat.imshow(X, aspect="auto", vmin=vmin, vmax=vmax)

    if annotate:
        for i1, x1 in enumerate(X):
            for i2, x2 in enumerate(X):
                y = X[i1, i2]
                if vmin is not None and vmax is not None:
                    if y > (vmax - vmin) / 2 + vmin:
                        color = "black"
                    else:
                        color = "white"
                else:
                    if y >= 0.5 * X.mean():
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
        "res.append(Residue(model, int(index), int(resi), chain, resn, oneletter, conservation))",
        space={"res": res, "Residue": Residue},
    )
    return res[0]


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


@new_command
def super_clustal_align(mobile: str, target: str, conservation: str = '*:.'):
    omega = clustal_omega(f"{mobile} | {target}", conservation)
    target_sele = None
    for obj, (atoms) in omega.items():
        ixs = '+'.join(str(at.index) for at in atoms)
        sele = f"%{obj} & index {ixs}"
        if not target_sele:
            target_sele = sele
            continue 
        pm.super(sele, target_sele)


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