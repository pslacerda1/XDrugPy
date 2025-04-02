import subprocess
import os
import atexit
import numpy as np
from shutil import rmtree
from tempfile import mkdtemp
from pymol import Qt
from os.path import exists


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
    if X.ndim == 1:
        X = squareform(X)
    Z = sch.linkage(X, method=method)
    if 'color_threshold' not in kwargs or kwargs['color_threshold'] < 0:
        kwargs['color_threshold'] = 0.7 * max(Z[:,2])

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
    for color, leaves in groups.items():
        for i1, leaf1 in enumerate(leaves):
            sum_dists = 0
            for i2 in range(len(leaves)):
                if i1 >= i2:
                    continue
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
    plt.tight_layout()
    plt.show()
    return dendro