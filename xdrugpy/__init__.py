from os.path import exists
import os
import stat
import platform
from urllib.request import urlretrieve

from .utils import run, RESOURCES_DIR


#
# INSTALL VINA
#

system = platform.system().lower()

match system:
    case "windows":
        bin_fname = "vina_1.2.5_win.exe"
    case "linux":
        bin_fname = "vina_1.2.5_linux_x86_64"
    case "darwin":
        bin_fname = "vina_1.2.5_mac_x86_64"
vina_url = f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/{bin_fname}"
vina_exe = f"{RESOURCES_DIR}/vina"
if system == "windows":
    vina_exe += ".exe"
if not exists(vina_exe):
    print(f'Installing AutoDock Vina on', vina_exe)
    urlretrieve(vina_url, vina_exe)
    os.chmod(vina_exe, stat.S_IEXEC)


#
# INSTALL MORE REQUIREMENTS
#

try:
    import numpy, pandas, scipy, matplotlib, seaborn, openpyxl, plip, rdkit, lxml, openbabel, strenum
except ImportError:
    run(
        "conda install -y"
        " matplotlib openpyxl scipy meeko plip openbabel rdkit pandas lxml strenum"
    )
try:
    import scrubber, meeko
except ImportError:
    run(
        "pip --disable-pip-version-check install"
        " https://github.com/pslacerda/molscrub/archive/refs/heads/windows.exe.zip"
        " https://github.com/forlilab/Meeko/archive/refs/tags/v0.6.1.zip"
    )


#
# INITIALIZE THE PLUGIN
#

def __init_plugin__(app=None):
    print("This version of XDrugPy is intended for non-comercial and academic purposes only.")
    from pymol import cmd
    cmd.undo_disable()
    if system == "windows":
        os.environ['PATH'] = "%s;%s" % (RESOURCES_DIR, os.environ['PATH'])
    else:
        os.environ['PATH'] = "%s:%s" % (RESOURCES_DIR, os.environ['PATH'])
    from .hotspots import __init_plugin__ as __init_hotspots__
    from .docking import __init_plugin__ as __init_docking__
    __init_hotspots__()
    __init_docking__()

if __name__ in ["pymol", "pmg_tk.startup.XDrugPy"]:
    __init_plugin__()