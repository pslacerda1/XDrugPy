import pytest
from pymol import cmd as pm

from xdrugpy.hotspots import (
    load_ftmap,
    calc_multivariate_hca,
    calc_univariate_hca,
    calc_fingerprints,
    get_fo,
    get_dce,
    get_dco,
    LinkageMethod,
    UnivariateMethod,
)

from . import images_identical, ResultFigures, PKG_DATA_DIR

@pytest.fixture
def test_name(request):
    yield request.function.__name__

@pytest.fixture(scope='module')
def load_default_1dq8():
    ftmap_1dq8 = load_ftmap(
        filename=PKG_DATA_DIR / "1dq8_atlas.pdb",
        group="default_1dq8",
        deep_search=False
    )
    yield ftmap_1dq8
    pm.delete('default_1da8')

@pytest.fixture(scope='module')
def load_default_1dq9():
    ftmap_1dq8 = load_ftmap(
        filename=PKG_DATA_DIR / "1dq9_atlas.pdb",
        group="default_1dq9",
        deep_search=False
    )
    yield ftmap_1dq8
    pm.delete('default_1da9')

@pytest.fixture(scope='module')
def load_deep_1dq9():
    ftmap_1dq9 = load_ftmap(
        filename=PKG_DATA_DIR / "1dq9_atlas.pdb",
        group="deep_1dq9",
        deep_search=True,
        remove_nested=True,
    )
    yield ftmap_1dq9
    pm.delete('deep_1da9')

@pytest.fixture(scope='module')
def load_deep_1dq8():
    ftmap_1dq8 = load_ftmap(
        filename=PKG_DATA_DIR / "1dq8_atlas.pdb",
        group="deep_1dq8",
        deep_search=True,
        remove_nested=True,
    )
    yield ftmap_1dq8
    pm.delete('deep_1da8')

@pytest.fixture(scope='module')
def load_deep_2tpr():
    ftmap = load_ftmap(
        filename=f"{PKG_DATA_DIR}/2TPR.pdb",
        group='deep_2tpr',
        deep_search=True,
        remove_nested=True,
    )
    yield ftmap
    pm.delete('deep_2tpr')



def test_calc_multivariate_hca(
    load_default_1dq8,
    load_default_1dq9
):
    figs = ResultFigures('test_calc_multivariate_hca')

    *_, medoids = calc_multivariate_hca(
        sele="(default_1dq8.CS.* OR default_1dq9.CS.*) AND p.S>13",
        color_threshold=2,
        annotate=True,
        linkage_method=LinkageMethod.WARD,
        dendrogram_plot=figs.generated,
    )
    assert medoids["C1"].pop() in ["default_1dq8.CS.0", "default_1dq9.CS.0"]
    assert medoids["C1"].pop() in ["default_1dq8.CS.0", "default_1dq9.CS.0"]
    assert len(medoids["C1"]) == 0

    assert images_identical(figs.generated, figs.reference)


def test_calc_univariate_hca_fo(
    load_deep_1dq9,
    load_deep_1dq8,
    load_deep_2tpr
):
    dendro_figs = ResultFigures("test_calc_univariate_hca_fo_dendro")
    heat_figs = ResultFigures("test_calc_univariate_hca_fo_heat")

    calc_univariate_hca(
        sele="deep_1dq8.DL.* OR deep_1dq9.DL.* OR deep_2tpr.DL.*",
        dist_method=UnivariateMethod.FO_AVG,
        linkage_method=LinkageMethod.AVERAGE,
        only_medoids=False,
        radius=4,
        annotate=False,
        nclusters=3,
        dendrogram_plot=dendro_figs.generated,
        heatmap_plot=heat_figs.generated,
    )
    assert images_identical(dendro_figs.generated, dendro_figs.reference)
    assert images_identical(heat_figs.generated, heat_figs.reference)


def test_calc_univariate_hca_jaccard(
    test_name,
    load_deep_1dq8,
    load_deep_1dq9
):
    dendro_figs = ResultFigures(f"{test_name}_dendro")
    heat_figs = ResultFigures(f"{test_name}_heat")

    calc_univariate_hca(
        sele="deep_1dq8.DL.* OR deep_1dq0.DL.*",
        dist_method=UnivariateMethod.JACCARD,
        linkage_method=LinkageMethod.AVERAGE,
        only_medoids=False,
        radius=4,
        annotate=True,
        nclusters=5,
        dendrogram_plot=dendro_figs.generated,
        heatmap_plot=heat_figs.generated,
    )
    assert images_identical(dendro_figs.generated, dendro_figs.reference)
    assert images_identical(heat_figs.generated, heat_figs.reference)


def test_overlap():
    pm.fetch('1OD')
    pm.fetch('NH2')
    assert get_fo("%NH2", "%1OD", radius=3.0) == 1.0
    assert round(get_dce("%NH2", "%1OD", radius=3.0), 2) == 7.67
    assert round(get_dco("%NH2", "%1OD", radius=3.0), 2) == 0.10
    assert get_dce("NotFound", "%NH2") == 0


def test_calc_fingerprint(test_name, load_deep_1dq8):

    fpt_figs = ResultFigures(test_name)
    calc_fingerprints(
        "deep_1dq8.CS.0 / deep_1dq8.CS.3",
        site="deep_1dq8.CS.0 | deep_1dq8.CS.3",
        site_radius=4,
        sharex=True,
        share_ylim=True,
        fingerprints_plot=fpt_figs.generated,
        nbins=50,
        heatmap_plot=False,
        dendrogram_plot=False,
    )
    assert images_identical(fpt_figs.generated, fpt_figs.reference)


def test_calc_fingerprint_clustering(
    test_name,
    load_deep_1dq8,
    load_deep_1dq9
):
    fpt_figs = ResultFigures(f"{test_name}_fpt")
    dendro_figs = ResultFigures(f"{test_name}_dendro")

    calc_fingerprints(
        multi_seles="deep_1dq8.CS* OR deep_1dq8.D* / deep_1dq9.B* | deep_1dq9.D*",
        site='chain B',
        site_radius=0.0,
        contact_radius=4.0,
        nbins=50,
        sharex=False,
        share_ylim=False,
        fingerprints_plot=fpt_figs.generated,
        dendrogram_plot=dendro_figs.generated,
    )
    assert images_identical(fpt_figs.generated, fpt_figs.reference)
    assert images_identical(dendro_figs.generated, dendro_figs.reference)


def test_load():
    pm.reinitialize()

    ftmap = load_ftmap(
        f"{PKG_DATA_DIR}/2TPR.pdb",
        deep_search=True,
        remove_nested=False,
    )
    hotspots = ftmap.hotspots
    assert len(hotspots) == 42
    assert hotspots[0].Object == '2TPR.DS.0'

    ftmap = load_ftmap(
        f"{PKG_DATA_DIR}/1dqa_atlas.pdb",
        "1dqa",
        deep_search=False,
    )
    assert len(ftmap.hotspots) == 1

    ftmap = load_ftmap(
        f'{PKG_DATA_DIR}/3mer_c10.pdb'
    )
    assert len(ftmap.cavities) == 2
    assert len(ftmap.clusters) == 4
    assert len(ftmap.hotspots) == 4
    assert len(ftmap.eclusters) == 0


def test_load_eftmap():
    pm.reinitialize()

    ftmap = load_ftmap(
        PKG_DATA_DIR / 'p38_MAPK_1R39_pharm.pdb',
        "1R39",
    )
    assert len(ftmap.eclusters) == 43
    assert ftmap.eclusters[37].ProbeType == 'apolar'
    assert ftmap.eclusters[37].S == 104
