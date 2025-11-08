from os.path import exists
import os
import stat
from urllib.request import urlretrieve
import platform

from pymol import cmd as pm
from .utils import run_system, RESOURCES_DIR


SYSTEM = platform.system().lower()


def install_executables():
    #
    # INSTALL VINA
    #

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

    #
    # INSTALL FPOCKET
    #
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


def install_pip_packages():
    #
    # INSTALL MORE REQUIREMENTS
    #
    try:
        import meeko, lxml, pandas, openpyxl, seaborn, scipy, matplotlib, strenum, openbabel, rcsbapi, rich, watchdog, molscrub, rdkit
    except ImportError:
        run_system(
            "pip install"
            " pandas openpyxl scipy matplotlib strenum openbabel-wheel rcsb-api rich watchdog molscrub"
            " https://github.com/pslacerda/Meeko/archive/refs/heads/patch-1.zip"
        )

@pm.extend
def install_xdrugpy_requirements():
    install_executables()
    install_pip_packages()



#
# INITIALIZE THE PLUGIN
#
if SYSTEM == "windows":
    os.environ["PATH"] = "%s;%s" % (RESOURCES_DIR, os.environ["PATH"])
    if not exists("{RESOURCES_DIR}/vina.exe"):
        install_xdrugpy_requirements()
else:
    os.environ["PATH"] = "%s:%s" % (RESOURCES_DIR, os.environ["PATH"])
    if not exists("{RESOURCES_DIR}/vina"):
        install_xdrugpy_requirements()


def __init_plugin__(app=None):
    print(
        "This version of XDrugPy is intended for non-comercial and academic purposes only."
    )
    from .hotspots import __init_plugin__ as __init_hotspots__
    from .docking import __init_plugin__ as __init_docking__
    from .multi import __init_plugin__ as __init_multi__

    __init_hotspots__()
    __init_docking__()
    __init_multi__()

    pm.undo_disable()


if __name__ in ["pymol", "pmg_tk.startup.XDrugPy"]:
    __init_plugin__()
