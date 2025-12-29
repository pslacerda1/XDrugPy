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

    try:
        import pyKVFinder
    except:
        os.system("pip install --no-deps pyKVFinder")

    #
    # Install Vina
    #
    match SYSTEM:
        case "windows":
            bin_fname = "vina_1.2.7_win.exe"
        case "linux":
            bin_fname = "vina_1.2.7_linux_x86_64"
        case "darwin":
            bin_fname = "vina_1.2.7_mac_x86_64"
    vina_url = f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/{bin_fname}"
    vina_exe = RESOURCES_DIR / "vina"
    if SYSTEM == "windows":
        vina_exe += ".exe"
    if not exists(vina_exe):
        print(f"Installing AutoDock Vina on", RESOURCES_DIR)
        urlretrieve(vina_url, vina_exe)
        os.chmod(vina_exe, stat.S_IEXEC)

    #
    # Install Clustal Omega
    #
    if SYSTEM == "windows":
        web_name = "clustal-omega-1.2.2-win64.zip"
        local_name = RESOURCES_DIR / web_name
        if not exists(local_name):
            urlretrieve(
                f"https://github.com/pslacerda1/XDrugPy/raw/refs/heads/master/bin/{web_name}",
                local_name
            )
            import zipfile
            zipfile.ZipFile(local_name).extractall(RESOURCES_DIR)
            os.chmod(RESOURCES_DIR / 'clustalo.exe', stat.S_IEXEC)
    else:
        if SYSTEM == "linux":
            web_name = "clustalo-1.2.4-Ubuntu-x86_64"
        elif SYSTEM == "darwin":
            web_name = "clustal-omega-1.2.3-macosx"
        else:
            raise RuntimeError("Unexpected system.")
        
        clustalo_url = f"https://github.com/pslacerda1/XDrugPy/raw/refs/heads/master/bin/{web_name}"
        clustalo_exe = RESOURCES_DIR / "clustalo"
        if not exists(clustalo_exe):
            urlretrieve(clustalo_url, clustalo_exe)
            os.chmod(clustalo_exe, stat.S_IEXEC)


def __init_plugin__(app=None):
    from .hotspots import __init_plugin__ as __init_hotspots__
    from .docking import __init_plugin__ as __init_docking__
    from .multi import __init_plugin__ as __init_multi__

    __init_hotspots__()
    __init_docking__()
    __init_multi__()

    pm.undo_disable()
    
    from matplotlib import pyplot as plt
    plt.rcParams.update({
        'font.size': 14,
        'figure.figsize': (10, 6),
        'svg.fonttype': 'none'
    })


if SYSTEM == "windows":
    os.environ["PATH"] = "%s;%s" % (RESOURCES_DIR, os.environ["PATH"])
else:
    os.environ["PATH"] = "%s:%s" % (RESOURCES_DIR, os.environ["PATH"])


if __name__ in ["pymol", "pmg_tk.startup.XDrugPy"]:
    __init_plugin__()
