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


def __init_plugin__():
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

    from .docking import __init_plugin__ as __init_docking__
    from .hotspots import __init_plugin__ as __init_hotspots__
    from .multi import __init_plugin__ as __init_multi__

    __init_hotspots__()
    __init_docking__()
    __init_multi__()


from .hotspots import (
    load_ftmap, get_fo, get_dc, get_dce,
    calc_multivariate_hca, calc_univariate_hca, calc_overlap_matrix,
    calc_fingerprints,
    LinkageMethod, OverlapFunction, UnivariateMethod, MultivariateDistanceMethod
)
from .commons import configure_matplotlib