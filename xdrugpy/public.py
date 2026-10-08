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

    # commons
    "configure_matplotlib",
]


def __init_plugin__(app=None):
    from .commons import configure_matplotlib

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
    
    from pymol import Qt
    QLocale = Qt.QtCore.QLocale

    QLocale.setDefault(QLocale("en_US"))

    from .hotspots import __init_plugin__ as __init_hotspots__
    from .docking import __init_plugin__ as __init_docking__
    from .multi import __init_plugin__ as __init_multi__

    __init_hotspots__()
    __init_docking__()
    __init_multi__()
    
    from textwrap import dedent
    from .commons import VERSION_FILE

    version_sha, version_date = VERSION_FILE.read_text().strip().splitlines()
    print(dedent(f"""
        XDrugPy pre-release candidate
         Cite the old DOI:  https://doi.org/10.1007/s10822-021-00403-8
            Github commit:  {version_sha}
              Commit date:  {version_date}
    """))


try:
    from .hotspots import (
        load_ftmap, get_fo, get_dc, get_dce,
        calc_multivariate_hca, calc_univariate_hca, calc_overlap_matrix,
        calc_fingerprints,
        LinkageMethod, OverlapFunction, UnivariateMethod, MultivariateDistanceMethod
    )
    from .commons import configure_matplotlib
except ImportError as exc:
    import traceback
    traceback.print_exc()
