import subprocess
import os
import atexit
import re
import numpy as np
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

QStandardPaths = Qt.QtCore.QStandardPaths


RESOURCES_DIR = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
if not exists(RESOURCES_DIR):
    os.makedirs(RESOURCES_DIR)

LIGAND_LIBRARIES_DIR = RESOURCES_DIR + '/libs/ligands/'
if not exists(LIGAND_LIBRARIES_DIR):
    os.makedirs(LIGAND_LIBRARIES_DIR)

RECEPTOR_LIBRARIES_DIR = RESOURCES_DIR + '/libs/receptors/'
if not exists(RECEPTOR_LIBRARIES_DIR):
    os.makedirs(RECEPTOR_LIBRARIES_DIR)

TEMPDIR = mkdtemp(prefix='XDrugPy-')
def clear_temp():
    rmtree(TEMPDIR)
atexit.register(clear_temp)


Residue = namedtuple("Reisude", "model index resn resi chain x y z")


ONE_LETTER = {
    "VAL": "V",
    "ILE": "I",
    "LEU": "L",
    "GLU": "E",
    "GLN": "Q",
    "ASP": "D",
    "ASN": "N",
    "HIS": "H",
    "TRP": "W",
    "PHE": "F",
    "TYR": "Y",
    "ARG": "R",
    "LYS": "K",
    "SER": "S",
    "THR": "T",
    "MET": "M",
    "ALA": "A",
    "GLY": "G",
    "PRO": "P",
    "CYS": "C",
}


def run(command, log=True, cwd=None, env=os.environ):
    if log:
        print('RUNNING PROCESS:', command)            
    ret = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        shell=True,
        env=env,
    )
    output = ret.stdout.decode(errors='replace')
    success = ret.returncode == 0
    return output, success


def run_system(command):
    print('RUNNING SYSTEM PROCESS:', command)
    os.system(command)

def dendrogram(X, method='ward', ax=None, **kwargs):
    from scipy.spatial.distance import squareform
    X = np.array(X)
    if X.ndim == 1:
        X = squareform(X)

    ax_file = False
    if ax is None:
        fig, ax = plt.subplots()
    elif isinstance(ax, str):
        ax_file = ax
        fig, ax = plt.subplots()
    
    Z = sch.linkage(X, method=method)
    dendro = dendrogram_linked(Z, **kwargs)
    if kwargs.get('orientation') == 'right':
        axline = ax.axvline
        ticklabels = ax.get_yticklabels()
    elif kwargs.get('orientation') == 'top':
        axline = ax.axhline
        ticklabels = ax.get_xticklabels()
    
    axline(kwargs["color_threshold"], color="gray", ls='--')
    groups = {}
    for color, leaf in zip(dendro['leaves_color_list'], dendro['ivl']):
        if color not in groups:
            groups[color] = []
        groups[color].append(leaf)
    dists_sum = {}
    for color, leaves in groups.items():
        for i1, leaf1 in enumerate(leaves):
            sum_dists = 0
            for i2 in range(len(leaves)):
                d = X[i1, i2]
                sum_dists += d
            dists_sum[(color, leaf1)] = sum_dists
    medoids = {}
    min_color = None
    for (color, leaf), rms in dists_sum.items():
        if color != min_color:
            min_rms = float('inf')
            min_color = color
        if rms < min_rms:
            min_rms = rms
            medoids[min_color] = leaf
    for label in ticklabels:
        for color, leaf in medoids.items():
            if label.get_text() == leaf:
                label.set_color(color)
    if ax_file:
        # plt.tight_layout()
        plt.savefig(ax_file)
    return dendro

def dendrogram_linked(Z, labels=None, ax=None, **kwargs):
    from matplotlib import pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    if 'color_threshold' not in kwargs or kwargs['color_threshold'] < 0:
        kwargs['color_threshold'] = 0.7 * max(Z[:,2])

    dendro = sch.dendrogram(
        Z,
        labels=labels,
        ax=ax,
        **kwargs
    )
    return dendro


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
        plt.tight_layout()
        plt.show()
    elif isinstance(ax, (str, Path)):
        plt.tight_layout()
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
                match = re.match(r'(Class|S|S0|S1|CD|MD|Lenght|Fpocket)(>=|<=|!=|==|=|>|<)(.*)', part)
                if match:
                    atom_data = {}
                    pm.iterate(
                        obj,
                        dedent("""
                            atom_data[model] = {
                                'Type': p.Type, 'Class':p.Class, 'S':p.S, 'S0':p.S0, 'S1':p.S1, 'CD':p.CD, 'MD':p.MD
                            }
                        """),
                        space={"atom_data": atom_data}
                    )
                    no_data = not atom_data or obj not in atom_data or atom_data[obj]['Type'] is None
                    if no_data:
                        continue
                    if type is not None and type != atom_data[obj]['Type']:
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
                    if op == '=':
                        op = '=='
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
                if pm.get_property('Type', obj) == type:
                    objects1.add(obj)
    if not objects2:
        objects2 = objects1
    if objects2 and not objects1:
        return objects2
    else:
        objects = (objects1.intersection(objects2))
    return objects


def multiple_expression_selector(exprs, type=None):
    object_list = []
    for expr in exprs.split(':'):
        object_list.append(expression_selector(expr, type=type))
    return object_list



def plot_hca_base(X, labels, linkage_method, color_threshold, axis):
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 1], wspace=0.01, hspace=0.01)
    ax_dend_top = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    Z = linkage(X, method=linkage_method)
    dendro1 = dendrogram_linked(
        Z,
        labels=labels,
        orientation='top',
        color_threshold=color_threshold,
        leaf_rotation=90,
        ax=ax_dend_top,
        count_sort='ascending',
        no_labels=True,
    )
    
    ax_dend_top.axhline(color_threshold, color="gray", ls='--')

    X = np.array(X)
    if X.ndim == 1:
        X = distance.squareform(X)

    X = X[dendro1['leaves'], :]
    X = X[:, dendro1['leaves']]

    ax_heat.set_xticks(range(len(dendro1['ivl'])), dendro1['ivl'])
    ax_heat.set_yticks(range(len(dendro1['ivl'])), dendro1['ivl'])
    ax_heat.tick_params(axis='x', rotation=90)
    ax_heat.yaxis.tick_right()
    ax_heat.imshow(X, aspect='auto')

    visited = set()
    min_dists = defaultdict(list)
    colors = set(dendro1["leaves_color_list"])
    for color in colors:
        for leaf1_idx, leaf_label, color1 in zip(dendro1['leaves'], dendro1['ivl'], dendro1['leaves_color_list']):
            if color != color1:
                continue
            key = (leaf1_idx, leaf_label, color)
            # if key in visited:
            #     continue
            # visited.add(key)
            items = []
            for leaf2_idx, _ in zip(dendro1['leaves'], dendro1['ivl']):
                d = X[leaf1_idx, leaf2_idx]
                items.append((color, leaf_label, d))
                
            dists = {}
            d = 0 
            for color, _, d in items:
                d += d
            dists[(color, leaf_label)] = d
            dists = {c:d for c, d in sorted(dists.items(), key=lambda k: k[1])}
            new_items = []
            items = sorted(items, key=lambda k: k[2])
            value = items[-1][2]

            for color, leaf_label, dist in items:
                if dist == value:
                    new_items.append(key)
            min_dists[color] = [*min_dists[color], *new_items]
    
    medoids = {}
    for color in colors:
        min_d = float('inf')
        min_leaf = None
        for i1, leaf1, _ in min_dists[color]:
            d = 0
            for i2, _, _ in min_dists[color]:
                d += X[i1, i2]
            if d < min_d:
                min_d = d
                min_leaf = leaf1
        medoids[color] = min_d, min_leaf
    ticklabels = [
        *ax_heat.get_xticklabels(),
        *ax_heat.get_yticklabels()
    ]
    for color in colors:
        for label in ticklabels:
            d, leaf = medoids[color]
            if label.get_text() == leaf:
                label.set_color(color)
                label.set_fontstyle("italic")
    
    if isinstance(axis, (str, Path)):
        # TODO why this fail?
        # plt.tight_layout()
        plt.savefig(axis)
    elif axis is None:
        # TODO why this fail?
        # plt.tight_layout()
        plt.show()
    return dendro1, medoids