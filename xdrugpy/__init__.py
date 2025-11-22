from os.path import exists
import os
import stat
from urllib.request import urlretrieve
import platform
from pymol import cmd as pm
from .utils import RESOURCES_DIR

SYSTEM = platform.system().lower()


@pm.extend
def install_xdrugpy_requirements():
    try:
        import meeko, pandas, openpyxl, scipy, matplotlib, strenum, openbabel, rich, watchdog, molscrub, rdkit, pdb2pqr, propka, sklearn
    except:
        os.system(
            "pip install --upgrade"
            " pandas openpyxl scipy matplotlib strenum openbabel-wheel rcsb-api rich watchdog molscrub pdb2pqr propka scikit-learn"
            " https://github.com/pslacerda/Meeko/archive/refs/heads/patch-1.zip"
        )

    match SYSTEM:
        case "windows":
            bin_fname = "vina_1.2.7_win.exe"
        case "linux":
            bin_fname = "vina_1.2.7_linux_x86_64"
        case "darwin":
            bin_fname = "vina_1.2.7_mac_x86_64"
    vina_url = f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/{bin_fname}"
    vina_exe = f"{RESOURCES_DIR}/vina"
    if SYSTEM == "windows":
        vina_exe += ".exe"
    if not exists(vina_exe):
        print(f"Installing AutoDock Vina on", vina_exe)
        urlretrieve(vina_url, vina_exe)
        os.chmod(vina_exe, stat.S_IEXEC)

    match SYSTEM:
        case "windows":
            bin_fname = "fpocket.exe"
        case "linux" | "darwin":
            bin_fname = "fpocket"
    fpocket_exe = f"{RESOURCES_DIR}/{bin_fname}"
    if not exists(fpocket_exe):
        print(f"Installing Fpocket on", fpocket_exe)
        fpocket_url = f"https://github.com/pslacerda/XDrugPy/raw/refs/heads/master/bin/fpocket.{SYSTEM}"
        urlretrieve(fpocket_url, fpocket_exe)
        os.chmod(fpocket_exe, stat.S_IEXEC)


def __init_plugin__(app=None):
    from .hotspots import __init_plugin__ as __init_hotspots__
    from .docking import __init_plugin__ as __init_docking__
    from .multi import __init_plugin__ as __init_multi__

    __init_hotspots__()
    __init_docking__()
    __init_multi__()

    pm.undo_disable()


if SYSTEM == "windows":
    os.environ["PATH"] = "%s;%s" % (RESOURCES_DIR, os.environ["PATH"])
else:
    os.environ["PATH"] = "%s:%s" % (RESOURCES_DIR, os.environ["PATH"])


if __name__ in ["pymol", "pmg_tk.startup.XDrugPy"]:
    __init_plugin__()
