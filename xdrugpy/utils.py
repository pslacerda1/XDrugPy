import itertools
import subprocess
import os
import atexit
import re
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from collections import namedtuple, defaultdict
from shutil import rmtree
from tempfile import mkdtemp
from fnmatch import fnmatch
from textwrap import dedent
from pymol import Qt, cmd as pm, parsing
from os.path import exists
from contextlib import contextmanager
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage
from collections import namedtuple
from functools import lru_cache

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


Residue = namedtuple("Reisude", "model index resn resi chain x y z oneletter")


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


def run_system(command):
    print("RUNNING SYSTEM PROCESS:", command)
    os.system(command)


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
        raise Exception("Messy annotations")
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
            return function(*args, **kwargs)

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
        # plt.tight_layout()
        plt.show()
    elif isinstance(ax, (str, Path)):
        # plt.tight_layout()
        plt.savefig(output_file)


@declare_command
def align_groups(
    mobile_groups: Selection,
    target: Selection,
):
    for mobile in mobile_groups.split():
        pm.cealign(f"{mobile}.protein", f"{target}.protein")
        for inner in pm.get_object_list(f"{mobile}.*"):
            if ".protein" in inner:
                continue
            pm.matrix_copy(f"{mobile}.protein", inner)


def expression_selector(expr, type=None):
    objects = set()
    objects1 = set()
    objects2 = set()
    eq_true = set()
    eq_false = set()
    for part in expr.split():
        for obj in pm.get_names("objects"):
            if fnmatch(obj, part):
                objects1.add(obj)
            else:
                match = re.match(
                    r"(Class|S|S0|S1|CD|MD|Lenght|Fpocket)(>=|<=|!=|==|=|>|<)(.*)", part
                )
                if match:
                    atom_data = {}
                    pm.iterate(
                        obj,
                        dedent(
                            """
                            atom_data[model] = {
                                'Type': p.Type, 'Class':p.Class, 'S':p.S, 'S0':p.S0, 'S1':p.S1, 'CD':p.CD, 'MD':p.MD
                            }
                        """
                        ),
                        space={"atom_data": atom_data},
                    )
                    no_data = (
                        not atom_data
                        or obj not in atom_data
                        or atom_data[obj]["Type"] is None
                    )
                    if no_data:
                        continue
                    if type is not None and type != atom_data[obj]["Type"]:
                        continue

                    def convert_type(value):
                        try:
                            return int(value)
                        except:
                            try:
                                return float(value)
                            except:
                                return f"'{value}'"

                    op = match.groups()[1]
                    if op == "=":
                        op = "=="
                    prop = match.groups()[0]
                    value = match.groups()[2]
                    prop = convert_type(atom_data[obj][prop])
                    value = convert_type(value)
                    try:
                        if eval(f"{prop}{op}{value}"):
                            eq_true.add(obj)
                        else:
                            eq_false.add(obj)
                    except:
                        eq_false.add(obj)
    objects2 = eq_true.difference(eq_false)
    if not objects1 and not objects2:
        return set()
    if not objects1:
        if type is not None:
            for obj in pm.get_names("objects"):
                if pm.get_property("Type", obj) == type:
                    objects1.add(obj)
    if not objects2:
        objects2 = objects1
    if objects2 and not objects1:
        return objects2
    else:
        objects = objects1.intersection(objects2)
    return objects


def multiple_expression_selector(exprs, type=None):
    object_list = []
    for expr in exprs.split(":"):
        object_list.append(expression_selector(expr, type=type))
    return object_list


def plot_hca_base(dists, labels, linkage_method, color_threshold, hide_below_color_threshold, annotate, axis):
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 1], wspace=0.01, hspace=0.01)
    ax_dend_top = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    
    dists = np.array(dists)
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
    sims = 1 - dists
    np.fill_diagonal(sims, 1)

    X = sims
    X = X[dendro["leaves"], :]
    X = X[:, dendro["leaves"]]

    ax_heat.set_xticks(range(len(dendro["ivl"])), dendro["ivl"])
    ax_heat.set_yticks(range(len(dendro["ivl"])), dendro["ivl"])
    ax_heat.tick_params(axis="x", rotation=90)
    ax_heat.yaxis.tick_right()
    ax_heat.imshow(sims, aspect="auto", vmin=0, vmax=1)

    if annotate:
        for i1, x1 in enumerate(sims):
            for i2, x2 in enumerate(sims):
                y = sims[i1, i2]
                if y >= 0.5:
                    color = "black"
                else:
                    color = "white"
                label = f"{y:.2f}"
                ax_heat.text(i2, i1, label, color=color, ha="center", va="center")

    # Calcular a soma das distâncias para cada ponto do cluster
    min_dists = defaultdict(float)
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
                d_sum += X[leaf1_idx, leaf2_idx]
            
            min_dists[(color, leaf_label1)] = d_sum

    # Encontrar o medoid (ponto com MENOR soma de distâncias) para cada cluster
    medoids = {}
    for color in colors:
        min_d = float("inf")
        min_leaf = None
        
        for (color1, leaf_label1), d_sum in min_dists.items():
            if color1 != color:
                continue
            
            # Medoid é o ponto com a MENOR soma de distâncias
            if d_sum < min_d:
                min_d = d_sum
                min_leaf = leaf_label1
        
        if min_leaf:
            for (color1, leaf_label1), d_sum in min_dists.items():
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
    
    if hide_below_color_threshold and color_threshold > 0:
        omitted_labels = list(medoids.values())
        for label in ticklabels:
            if label.get_text() not in omitted_labels:
                label.set_visible(False)
    
    if not axis:
        plt.show()
    elif isinstance(axis, (str, Path)):
        plt.savefig(axis)
    return dendro, medoids


@lru_cache(999999999)
def get_residue_from_object(obj, idx):
    res = []
    pm.iterate_state(
        -1,
        f"%{obj} & index {idx}",
        "res.append(Residue(model, int(index), resn, int(resi), chain, float(x), float(y), float(z), oneletter))",
        space={"res": res, "Residue": Residue},
    )
    return res[0]


def get_mapping(
    polymers: Selection,
    site: str = "*",
    radius: float = 2,
):
    # Get polymers to be mapped to reference site
    ref_polymer = polymers[0]
    polymers = {p: True for p in polymers}  # ordered set

    # Do the alignmnet
    mappings = np.empty((0, 9))
    for polymer in polymers:
        try:
            aln_obj = pm.get_unused_name()
            pm.cealign(
                f"{ref_polymer} within {radius} from {site}",
                polymer,
                transform=False,
                object=aln_obj,
            )
            aln = pm.get_raw_alignment(aln_obj)
        finally:
            pm.delete(aln_obj)
        for (obj1, idx1), (obj2, idx2) in aln:
            res = get_residue_from_object(obj2, idx2)
            mappings = np.vstack([mappings, res])
    return pd.DataFrame(mappings, columns=Residue._fields)
