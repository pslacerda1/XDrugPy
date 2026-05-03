__ALL__ = [
    # hotspots
    "load_fstmap",
    "get_fo",
    "get_dc",
    "get_dce",
    "get_ho",
    "calc_multivariate_hca",
    "calc_overlap_matrix",
    "LinkageMethod",

    # utils
    "new_command",
    "ArgumentParsingError",
]


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

from .hotspots import load_ftmap, get_fo, get_dc, get_dce, get_ho, calc_multivariate_hca, calc_overlap_matrix, LinkageMethod
from .utils import ArgumentParsingError, new_command

if __name__ in ["pymol", "pmg_tk.startup.XDrugPy"]:
    __init_plugin__()

