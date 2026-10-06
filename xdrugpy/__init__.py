import sys
import json
import os
import platform
import shutil
import stat
import zipfile
from tempfile import mkdtemp
from pathlib import Path
import urllib.request
from urllib.request import urlretrieve
from subprocess import check_call, CalledProcessError
from textwrap import dedent
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
]


QStandardPaths = Qt.QtCore.QStandardPaths


try:
    data_location = QStandardPaths.AppLocalDataLocation
except AttributeError:
    try:
        data_location = QStandardPaths.AppDataLocation
    except AttributeError:
        data_location = QStandardPaths.StandardLocation.AppDataLocation

RESOURCES_DIR = Path(
    QStandardPaths.writableLocation(data_location)
) / "XDrugPy"

RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

LIGAND_LIBRARIES_DIR = Path(RESOURCES_DIR / "libs/ligands/")
LIGAND_LIBRARIES_DIR.mkdir(parents=True, exist_ok=True)

RECEPTOR_LIBRARIES_DIR = Path(RESOURCES_DIR / "libs/receptors/")
RECEPTOR_LIBRARIES_DIR.mkdir(parents=True, exist_ok=True)

TEMPDIR = Path(mkdtemp(prefix="XDrugPy-"))


PLUGIN_VERSION_DEFAULT = "master"
RUST_PROGRAM_VERSION = "v.40"

VERSION_FILE = Path(RESOURCES_DIR) / "version.txt"


@pm.extend
def xdrugpy_install(plugin_version=PLUGIN_VERSION_DEFAULT):

    # Record a version file
    github_repo_url = f"https://api.github.com/repos/pslacerda1/XDrugPy/commits/{plugin_version}"
    with urllib.request.urlopen(github_repo_url) as response:
        text = response.read().decode("utf-8")
    
    data = json.loads(text)
    version_sha = data['sha']
    version_date = data['commit']['committer']['date']

    VERSION_FILE.write_text(version_sha + '\n' + version_date)
    
    try:
        check_call([
            sys.executable, "-m", "pip", "install",
            f"https://github.com/pslacerda1/XDrugPy/archive/{version_sha}.zip",
        ])
        check_call([
            sys.executable, "-m", "pip", "install",
            "-r", f"http://raw.githubusercontent.com/pslacerda1/XDrugPy/{version_sha}/requirements.txt"
        ])
        check_call([
            sys.executable, "-m", "pip", "install", "numpy==1.26.4", "scipy==1.15.3"
        ])

        check_call([
            sys.executable, "-m", "pip", "install", "--no-deps",
            "https://github.com/pslacerda1/pymol_new_command/archive/refs/heads/main.zip"
        ])
        try:
            check_call([
                sys.executable, "-m", "pip", "install", "--no-deps",
                "pyKVFinder==0.9.5",
            ])
        except CalledProcessError as exc:
            print("Continuing without pyKVFinder.")

    except CalledProcessError as exc:
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
    urlretrieve(url, exe)
    os.chmod(exe, stat.S_IXUSR)

    #
    # Install Clustal Omega
    #
    match system:
        case "windows":
            web_name = "clustal-omega-1.2.2-win64.zip"
            local_zip = RESOURCES_DIR / web_name
            local_exe = RESOURCES_DIR / 'clustalo.exe'
            if local_zip.exists():
                os.unlink(local_zip)
            urlretrieve(
                f"https://github.com/pslacerda1/XDrugPy/raw/refs/heads/master/misc/{web_name}",
                local_zip
            )
            zipfile.ZipFile(local_zip).extractall(RESOURCES_DIR)
            for file in (RESOURCES_DIR / "clustal-omega-1.2.2-win64").glob("*"):
                shutil.move(file, RESOURCES_DIR)
            os.chmod(local_exe, stat.S_IXUSR)

        case "darwin" | "linux":
            check_call([
                'conda', 'install', '-y', 'bioconda::clustalo'
            ])

    #
    # Install My (alpha) Rust Project
    #
    match system:
        case "linux":
            web_name = "xdrugpy_xhf-ubuntu"
        case "windows":
            web_name = "xdrugpy_xhf-windows.exe"
        case "darwin":
            web_name = "xdrugpy_xhf-macos"
        case _:
            raise RuntimeError("Unexpected system.")
    url = f"https://github.com/pslacerda1/xdrugpy_xhf/releases/download/{RUST_PROGRAM_VERSION}/{web_name}"
    exe = RESOURCES_DIR / "xdrugpy_xhf"
    if system == "windows":
        exe = exe.with_suffix('.exe')
    if exe.exists():
        os.unlink(exe)
    urlretrieve(url, exe)
    os.chmod(exe, stat.S_IXUSR)


def __init_plugin__(app=None):
    from .utils import configure_matplotlib

    configure_matplotlib(
        style="default",
        backend="qtagg",
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

    version_sha, version_date = VERSION_FILE.read_text().strip().splitlines()
    print(dedent(f"""
        XDrugPy pre-release candidate
         Cite the old DOI:  https://doi.org/10.1007/s10822-021-00403-8
            Github commit:  {version_sha}
              Commit date:  {version_date}
    """))


os.environ["PATH"] = str(RESOURCES_DIR) + os.pathsep + os.environ["PATH"]
os.environ["PATH"] = str(RESOURCES_DIR) + "/PyMOL" + os.pathsep + os.environ["PATH"]

try:
    from .hotspots import (
        load_ftmap, get_fo, get_dc, get_dce,
        calc_multivariate_hca, calc_univariate_hca, calc_overlap_matrix,
        calc_fingerprints,
        LinkageMethod, OverlapFunction, UnivariateMethod, MultivariateDistanceMethod
    )
    from .utils import configure_matplotlib
except ImportError as exc:
    import traceback
    traceback.print_exc()
