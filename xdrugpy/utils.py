import subprocess
import os
import atexit
import re
import numpy as np
from collections import namedtuple
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


def dendrogram(X, labels=None, method='ward', ax=None, **kwargs):
    from scipy.spatial.distance import squareform
    import scipy.cluster.hierarchy as sch
    from matplotlib import pyplot as plt

    X = np.array(X)
    if ax is None:
        _, ax = plt.subplots()
    Z = sch.linkage(X, method=method)
    if 'color_threshold' not in kwargs or kwargs['color_threshold'] < 0:
        kwargs['color_threshold'] = 0.7 * max(Z[:,2])

    ax_file = False
    if ax is None:
        fig, ax = plt.subplots()
    elif isinstance(ax, str):
        ax_file = ax
        fig, ax = plt.subplots()
    
    dendro = sch.dendrogram(
        Z,
        labels=labels,
        ax=ax,
        **kwargs
    )
    if kwargs.get('orientation') == 'right':
        axline = ax.axvline
        ticklabels = ax.get_yticklabels()
    else:
        axline = ax.axhline
        ticklabels = ax.get_xticklabels()
        axline(kwargs["color_threshold"], color="gray", ls='--')
    groups = {}
    for color, leaf in zip(dendro['leaves_color_list'], dendro['ivl']):
        if color not in groups:
            groups[color] = []
        groups[color].append(leaf)
    dists_sum = {}
    if X.ndim == 1:
        X = squareform(X)
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
        plt.tight_layout()
        plt.savefig(ax_file)
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