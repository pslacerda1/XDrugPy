from os.path import exists
import os
import stat
import platform
from urllib.request import urlretrieve

from .utils import run_system, RESOURCES_DIR


#
# INSTALL VINA
#

system = platform.system().lower()

match system:
    case "windows":
        bin_fname = "vina_1.2.7_win.exe"
    case "linux":
        bin_fname = "vina_1.2.7_linux_x86_64"
    case "darwin":
        bin_fname = "vina_1.2.7_mac_x86_64"
vina_url = f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/{bin_fname}"
vina_exe = f"{RESOURCES_DIR}/vina"
if system == "windows":
    vina_exe += ".exe"
if not exists(vina_exe):
    print(f'Installing AutoDock Vina on', vina_exe)
    urlretrieve(vina_url, vina_exe)
    os.chmod(vina_exe, stat.S_IEXEC)

#
# INSTALL FPOCKET
#
match system:
    case 'windows':
        bin_fname = 'fpocket.exe'
    case 'linux' | 'darwin':
        bin_fname = 'fpocket'
fpocket_exe = f"{RESOURCES_DIR}/{bin_fname}"
if not exists(fpocket_exe):
    import os
    import stat
    from urllib.request import urlretrieve

    print(f'Installing Fpocket on', fpocket_exe)
    fpocket_url = f"https://github.com/pslacerda/XDrugPy/raw/refs/heads/master/bin/fpocket.{system}"
    urlretrieve(fpocket_url, fpocket_exe)
    os.chmod(fpocket_exe, stat.S_IEXEC)


#
# INSTALL MORE REQUIREMENTS
#
try:
    import scrubber, meeko, lxml, pandas, openpyxl, seaborn, scipy, matplotlib, strenum, openbabel
except ImportError:
    run_system(
        "pip install"
        " lxml pandas openpyxl seaborn scipy matplotlib strenum openbabel-wheel"
        " https://github.com/pslacerda/molscrub/archive/refs/heads/windows.exe.zip"
        " https://github.com/pslacerda/Meeko/archive/refs/heads/patch-1.zip"
    )

try:
    import plip, rdkit, openbabel
except ImportError:
    run_system(
        "conda install -y"
        " plip 'rdkit<2024'"
    )

#
# INITIALIZE THE PLUGIN
#
if system == "windows":
    os.environ['PATH'] = "%s;%s" % (RESOURCES_DIR, os.environ['PATH'])
else:
    os.environ['PATH'] = "%s:%s" % (RESOURCES_DIR, os.environ['PATH'])
    
def __init_plugin__(app=None):
    print("This version of XDrugPy is intended for non-comercial and academic purposes only.")
    from .hotspots import __init_plugin__ as __init_hotspots__
    from .docking import __init_plugin__ as __init_docking__
    from .rms import __init_plugin__ as __init_rmsf__
    __init_hotspots__()
    __init_docking__()
    __init_rmsf__()
    from pymol import cmd
    cmd.undo_disable()


if __name__ in ["pymol", "pmg_tk.startup.XDrugPy"]:
    __init_plugin__()