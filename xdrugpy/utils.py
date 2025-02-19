import subprocess
import os
import atexit
from shutil import rmtree
from tempfile import mkdtemp
from pymol import Qt
from os.path import exists

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


