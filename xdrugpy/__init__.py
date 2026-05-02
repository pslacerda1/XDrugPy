from os.path import exists
import os
from urllib.request import urlretrieve
from pymol import cmd as pm


@pm.extend
def install_xdrugpy_requirements():
    try:
        import pandas, matplotlib, scipy, strenum, networkx, openpyxl
    except:
        os.system(
            "pip install pandas matplotlib scipy strenum networkx openpyxl"
        )
    try:
        import pyKVFinder
    except:
        os.system("pip install --no-deps pyKVFinder")


def __init_plugin__(app=None):
    import matplotlib
    import matplotlib.style
    import matplotlib.colors
    from matplotlib import pyplot as plt
    from cycler import cycler
    
    matplotlib.use("Qt5Agg")
    matplotlib.style.use('default')
    plt.rcParams.update({
        'font.size': 14,
        'figure.figsize': (10, 6),
        'svg.fonttype': 'none',
        'axes.prop_cycle': cycler(color=reversed(matplotlib.colors.XKCD_COLORS))
    })

    from PyQt5.QtCore import QLocale
    QLocale.setDefault(QLocale("en_US"))

    from .hotspots import __init_plugin__ as __init_hotspots_plugin__
    __init_hotspots_plugin__()


if __name__ in ["pymol", "pmg_tk.startup.XDrugPy"]:
    __init_plugin__()
