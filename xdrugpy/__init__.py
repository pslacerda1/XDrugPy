import sys
from subprocess import check_call
from pymol import cmd as pm

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
    "EXPERIMENTAL_XDRUGPY"
]


@pm.extend
def xdrugpy_install():
    try:
        check_call([
            sys.executable, "-m", "pip", "install",
            "https://github.com/pslacerda1/XDrugPy/archive/refs/heads/DRUGpy_CAMLDDD.zip"
        ])
        check_call([
            sys.executable, "-m", "pip", "install", "--no-deps",
            "pyKVFinder==0.9.0",
        ])
    except Exception as e:
        print(f"XDrugPy: Installation failed: {e}")


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
    load_ftmap, get_fo, get_dc, get_dce, get_ho,
    calc_multivariate_hca, calc_univariate_hca, calc_overlap_matrix,
    LinkageMethod, OverlapFunction
)
from .utils import configure_matplotlib, plot, EXPERIMENTAL_XDRUGPY
