import sys
import os
import platform
import stat
import zipfile
from shutil import copyfile
from tempfile import mkdtemp
from pathlib import Path
from urllib.request import urlretrieve
from subprocess import check_call
from pymol import cmd as pm
from pymol import Qt


__ALL__ = [
    # hotspots
    "load_ftmap",
    "get_fo",
    "get_dc",
    "get_dce",
    "get_ho",
    "calc_multivariate_hca",
    "calc_univariate_hca",
    "calc_overlap_matrix",
    "calc_ligand_fit",
    "calc_fingerprints",
    "LinkageMethod",
    "OverlapFunction",
    "BindMetric",

    # utils
    "configure_matplotlib",
    "plot",
]


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


@pm.extend
def xdrugpy_install():
    try:
        check_call([
            sys.executable, "-m", "pip", "install",
            "https://github.com/pslacerda1/XDrugPy/archive/refs/heads/master.zip"
        ])
        check_call([  ## pyproject.toml --no-deps limitation
            sys.executable, "-m", "pip", "install", "--no-deps", "pyKVFinder==0.9.0",
        ])
    except Exception as e:
        print(f"XDrugPy: Installation failed: {e}")

    #
    # Install Vina
    #
    system = platform.system().lower()
    match system:
        case "windows":
            web_name = "vina_1.2.7_win.exe"
        case "linux":
            web_name = "vina_1.2.7_linux_x86_64"
        case "darwin":
            web_name = "vina_1.2.7_mac_x86_64"
    url = f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/{web_name}"
    exe = RESOURCES_DIR / 'vina'
    if system == "windows":
        exe += ".exe"
    if exe.exists():
        os.unlink(exe)
    print(f"Downloading {url} into {exe}")
    urlretrieve(url, exe)
    os.chmod(exe, stat.S_IRUSR | stat.S_IXUSR)

    #
    # Install MUSCLE Aignment
    #
    if system == "linux":
        web_name = "muscle-linux-x86.v5.3"
    elif system == "windows":
        web_name = " muscle-win64.v5.3.exe "
    elif system == "darwin":
        web_name = "muscle-osx-x86.v5.3"
    else:
        raise RuntimeError("Unexpected system.")
    url = f"https://github.com/rcedgar/muscle/releases/download/v5.3/{web_name}"
    exe = RESOURCES_DIR / "muscle"
    if system == "windows":
        exe += ".exe"
    if exe.exists():
        os.unlink(exe)
    print(f"Downloading {url} into {exe}")
    urlretrieve(url, exe)
    os.chmod(exe, stat.S_IRUSR | stat.S_IXUSR)

    #
    # Install My (alpha) Rust Project
    #
    rust_tag = "v.26"
    if system == "windows":
        web_name = "xdrugpy_hotspot_finder-windows.exe"
        exe = RESOURCES_DIR / "xdrugpy_hotspot_finder.exe"
        if exe.exists():
            os.unlink(exe)
        print(f"Downloading {url} into {exe}")
        urlretrieve(url, exe)
        os.chmod(exe, stat.S_IRUSR | stat.S_IXUSR)
    else:
        if system == "linux":
            web_name = "xdrugpy_hotspot_finder-ubuntu"
        elif system == "windows":
            web_name = "xdrugpy_hotspot_finder-windows.exe"
        elif system == "darwin":
            web_name = "xdrugpy_hotspot_finder-macos"
        else:
            raise RuntimeError("Unexpected system.")
        
        url = f"https://github.com/pslacerda1/xdrugpy_hotspot_finder/releases/download/{rust_tag}/{web_name}"
        exe = RESOURCES_DIR / "xdrugpy_hotspot_finder"
        if system == "windows":
            exe += ".exe"
        if exe.exists():
            os.unlink(exe)
        print(f"Downloading {url} into {exe}")
        urlretrieve(url, exe)
        os.chmod(exe, stat.S_IRUSR | stat.S_IXUSR)


def __init_plugin__(app=None):
    from .utils import configure_matplotlib

    configure_matplotlib("default", {
        'font.size': 14,
        'figure.figsize': (10, 6),
        'svg.fonttype': 'none',
        # 'axes.prop_cycle': cycler(color=reversed(matplotlib.colors.XKCD_COLORS))
    })

    from PyQt5.QtCore import QLocale
    QLocale.setDefault(QLocale("en_US"))

    from .hotspots import __init_plugin__ as __init_hotspots__
    from .docking import __init_plugin__ as __init_docking__
    from .multi import __init_plugin__ as __init_multi__

    __init_hotspots__()
    __init_docking__()
    __init_multi__()

    from textwrap import dedent
    print(dedent("""
        DRUGpy version 3.0 (a.k.a. Newer and Faster).
            Please read and cite: http://doi.com.br
    """))

os.environ["PATH"] = str(RESOURCES_DIR) + os.pathsep + os.environ["PATH"]
os.environ["PATH"] = str(RESOURCES_DIR) + "/PyMOL" + os.pathsep + os.environ["PATH"]

from .hotspots import (
    load_ftmap, get_fo, get_dc, get_dce,
    calc_multivariate_hca, calc_univariate_hca, calc_overlap_matrix,
    calc_fingerprints,
    LinkageMethod, OverlapFunction
)
from .utils import configure_matplotlib, plot, run
