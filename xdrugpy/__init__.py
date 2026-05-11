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
    "LinkageMethod",
    "OverlapFunction",
    "BindMetric",

    # utils
    "ArgumentParsingError",
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
            "https://github.com/pslacerda1/XDrugPy/archive/refs/heads/DRUGpy_CAMLDDD.zip"
        ])
        check_call([  ## pyproject.toml --no-deps limitation
            sys.executable, "-m", "pip", "install", "--no-deps", "pyKVFinder==0.9.0",
        ])
    except Exception as e:
        print(f"XDrugPy: Installation failed: {e}")

    #
    # Install Vina
    #
    system = platform.system()
    match system:
        case "windows":
            bin_fname = "vina_1.2.7_win.exe"
        case "linux":
            bin_fname = "vina_1.2.7_linux_x86_64"
        case "darwin":
            bin_fname = "vina_1.2.7_mac_x86_64"
    vina_url = f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/{bin_fname}"
    vina_exe = Path(f'{RESOURCES_DIR}/vina')
    if system == "windows":
        vina_exe += ".exe"
    if not vina_exe.exists():
        print(f"Installing AutoDock Vina on", RESOURCES_DIR)
        urlretrieve(vina_url, vina_exe)
        os.chmod(vina_exe, stat.S_IEXEC)

    #
    # Install Clustal Omega
    #
    if system == "windows":
        web_name = "clustal-omega-1.2.2-win64.zip"
        local_name = RESOURCES_DIR / web_name
        if not Path(local_name).exists():
            urlretrieve(
                f"https://github.com/pslacerda1/XDrugPy/raw/refs/heads/master/bin/{web_name}",
                local_name
            )
            import zipfile
            zipfile.ZipFile(local_name).extractall(RESOURCES_DIR)
            os.chmod(RESOURCES_DIR / 'clustalo.exe', stat.S_IEXEC)
    else:
        if system == "linux":
            web_name = "clustalo-1.2.4-Ubuntu-x86_64"
        elif system == "darwin":
            web_name = "clustal-omega-1.2.3-macosx"
        else:
            raise RuntimeError("Unexpected system.")
        
        clustalo_url = f"https://github.com/pslacerda1/XDrugPy/raw/refs/heads/master/bin/{web_name}"
        clustalo_exe = RESOURCES_DIR / "clustalo"
        if not clustalo_exe.exists():
            urlretrieve(clustalo_url, clustalo_exe)
            os.chmod(clustalo_exe, stat.S_IEXEC)



def __init_plugin__(app=None):
    from .utils import configure_matplotlib
    import matplotlib
    import matplotlib.style
    import matplotlib.colors
    from matplotlib import pyplot as plt
    from cycler import cycler

    configure_matplotlib("default", {
        'font.size': 14,
        'figure.figsize': (10, 6),
        'svg.fonttype': 'none',
        'axes.prop_cycle': cycler(color=reversed(matplotlib.colors.XKCD_COLORS))
    })

    from PyQt5.QtCore import QLocale
    QLocale.setDefault(QLocale("en_US"))

    from .hotspots import __init_plugin__ as __init_hotspots__
    # from .docking import __init_plugin__ as __init_docking__
    # from .multi import __init_plugin__ as __init_multi__

    __init_hotspots__()
    # __init_docking__()
    # __init_multi__()

    from textwrap import dedent
    print(dedent("""
        DRUGpy version 2.0 (a.k.a. DRUGpy_CAML_DDD).
            Please read and cite: http://doi.com.br
    """))


from .hotspots import (
    load_ftmap, get_fo, get_dc, get_dce,
    calc_multivariate_hca, calc_univariate_hca, calc_overlap_matrix,
    LinkageMethod, OverlapFunction
)
from .utils import configure_matplotlib, plot
