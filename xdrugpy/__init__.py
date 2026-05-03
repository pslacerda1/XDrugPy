import sys
from subprocess import check_call
from pymol import cmd as pm
import pymol_new_command


__ALL__ = [
    # hotspots
    "load_ftmap",
    "get_fo",
    "get_dc",
    "get_dce",
    "get_ho",
    "calc_multivariate_hca",
    "calc_overlap_matrix",
    "calc_ligand_fit",
    "LinkageMethod",
    "OverlapFunction",
    "BindMetric",

    # utils
    "new_command",
    "ArgumentParsingError",
    "configure_matplotlib",
    "plot",
]

try:
    import pyKVFinder
except ImportError:
    try:
        check_call([
            sys.executable, "-m", "pip", "install", "--no-deps",
            "pyKVFinder==0.9.0",
        ])
        check_call([
            sys.executable, "-m", "pip", "install", "/home/peu/Desktop/XDrugPy"
        ])
    except Exception as e:
        print(f"XDrugPy: Install pyKVFinder failed: {e}")


def __init_plugin__(app=None):
    from .utils import configure_matplotlib
    configure_matplotlib()

    from PyQt5.QtCore import QLocale
    QLocale.setDefault(QLocale("en_US"))

    from .hotspots import __init_plugin__ as __init_hotspots_plugin__
    __init_hotspots_plugin__()

    from textwrap import dedent
    print(dedent("""
        DRUGpy version 2.0 (a.k.a. DRUGpy_CAML_DDD).
            Please read and cite: http://doi.com.br
    """))


from .hotspots import (
    load_ftmap, get_fo, get_dc, get_dce, get_ho,
    calc_multivariate_hca, calc_overlap_matrix,
    LinkageMethod, OverlapFunction
)
from .utils import configure_matplotlib, plot
