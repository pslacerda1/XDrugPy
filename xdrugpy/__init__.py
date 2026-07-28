import sys
import os
import platform
import stat
from tempfile import mkdtemp
from pathlib import Path
from urllib.request import urlretrieve
from subprocess import check_call
from pymol import cmd as pm
from pymol import Qt


__ALL__ = [
    "xdrugpy_install",

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
    "DistanceMethod",
    "OverlapFunction",
    "HcaOverlapFunction",
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


XDRUGPY_PLUGIN_VERSION_DEFAULT = "heads/master"
XDRUGPY_PROGRAM_VERSION_DEFAULT = "v.37"


@pm.extend
def xdrugpy_install(
    plugin_version=XDRUGPY_PLUGIN_VERSION_DEFAULT,
    program_version=XDRUGPY_PROGRAM_VERSION_DEFAULT
):
    try:
        check_call([
            sys.executable, "-m", "pip", "install", "-U"
            f"https://github.com/pslacerda1/XDrugPy/archive/refs/{plugin_version}.zip"
        ])
        check_call([  ## pyproject.toml --no-deps limitation
            sys.executable, "-m", "pip", "install", "--no-deps", "pyKVFinder==0.9.0",
        ])
        check_call([
            'conda', 'install', '-y', 'bioconda::clustalo'
        ])
    except Exception as exc:
        raise SystemError(f"XDrugPy: Installation failed.") from exc

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
        case _:
            raise RuntimeError("Unexpected system.")

    url = f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/{web_name}"
    exe = RESOURCES_DIR / 'vina'
    if system == "windows":
        exe = exe.with_suffix('.exe')
    if exe.exists():
        os.unlink(exe)
    print(f"Downloading {url} into {exe}")
    urlretrieve(url, exe)
    os.chmod(exe, stat.S_IRUSR | stat.S_IXUSR)

    #
    # Install My (alpha) Rust Project
    #
    match system:
        case "linux":
            web_name = "xdrugpy_hotspot_finder-ubuntu"
        case "windows":
            web_name = "xdrugpy_hotspot_finder-windows.exe"
        case "darwin":
            web_name = "xdrugpy_hotspot_finder-macos"
        case _:
            raise RuntimeError("Unexpected system.")
    
    url = f"https://github.com/pslacerda1/xdrugpy_hotspot_finder/releases/download/{program_version}/{web_name}"
    exe = RESOURCES_DIR / "xdrugpy_hotspot_finder"
    if system == "windows":
        exe = exe.with_suffix('.exe')
    if exe.exists():
        os.unlink(exe)
    print(f"Downloading {url} into {exe}")
    urlretrieve(url, exe)
    os.chmod(exe, stat.S_IRUSR | stat.S_IXUSR)


def __init_plugin__(app=None):
    from .utils import configure_matplotlib

    configure_matplotlib(
        style="default",
        params={
        'font.size': 14,
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
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
        DRUGpy version 2.0a (a.k.a. Newer and Faster).
            Please read and cite: http://doi.com.br
    """))

os.environ["PATH"] = str(RESOURCES_DIR) + os.pathsep + os.environ["PATH"]
os.environ["PATH"] = str(RESOURCES_DIR) + "/PyMOL" + os.pathsep + os.environ["PATH"]

from .hotspots import (
    load_ftmap, get_fo, get_dc, get_dce,
    calc_multivariate_hca, calc_univariate_hca, calc_overlap_matrix,
    calc_fingerprints,
    LinkageMethod, OverlapFunction, UnivariateDistanceMethod, MultivariateDistanceMethod
)
from .utils import configure_matplotlib, plot, run
