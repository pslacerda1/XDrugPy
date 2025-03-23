import subprocess
import os
import atexit
import scipy.cluster.hierarchy as sch
import numpy as np
from scipy.spatial.distance import squareform
from shutil import rmtree
from tempfile import mkdtemp
from pymol import Qt
from os.path import exists
from matplotlib import pyplot as plt


QStandardPaths = Qt.QtCore.QStandardPaths


RESOURCES_DIR = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
if not exists(RESOURCES_DIR):
    os.makedirs(RESOURCES_DIR)

LIBRARIES_DIR = RESOURCES_DIR + '/libs/ligands/'
if not exists(LIBRARIES_DIR):
    os.makedirs(LIBRARIES_DIR)

MAPS_DIR = RESOURCES_DIR + '/libs/maps/'
if not exists(MAPS_DIR):
    os.makedirs(MAPS_DIR)

TEMPDIR = mkdtemp(prefix='runvina-')
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
