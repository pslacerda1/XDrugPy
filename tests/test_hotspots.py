import os.path
from pymol import cmd as pm
import numpy as np
import matplotlib as mpl
from xdrugpy.hotspots import (
    load_ftmap,
    calc_multivariate_hca,
    calc_univariate_hca,
    calc_fingerprints,
    res_sim,
    get_fo,
    get_dce,
    LinkageMethod,
    HcaOverlapFunction,
)

mpl.use('SVG')
mpl.rcParams['svg.hashsalt'] = 'fixed_salt_123'
mpl.rcParams['svg.fonttype'] = 'none'
np.random.seed(42)
pkg_data = os.path.dirname(__file__) + "/data"


def images_identical(img1_path, img2_path):
    from PIL import ImageChops, Image
    import io
    import cairosvg
    def rasterize(svg_path):
        png_data = cairosvg.svg2png(url=svg_path)
        return Image.open(io.BytesIO(png_data)).convert("RGB")
    def compare_visual(svg1, svg2):
        img1 = rasterize(svg1)
        img2 = rasterize(svg2)
        diff = ImageChops.difference(img1, img2)
        return diff.getbbox() is None  # True if identical
    return compare_visual(svg1=img1_path, svg2=img2_path)


def test_calc_multivariate_hca():
    pm.reinitialize()
    load_ftmap(f"{pkg_data}/1dq8_atlas.pdb", "1dq8")
    load_ftmap(f"{pkg_data}/1dq9_atlas.pdb", "1dq9")

    dendro_ref = f"{pkg_data}/test_calc_multivariate_hca_ref.svg"
    dendro_gen = f"{pkg_data}/test_calc_multivariate_hca_gen.svg"

    X, object_list, dendro, medoids = calc_multivariate_hca(
        sele="*.CS.* AND p.S>13",
        color_threshold=2,
        annotate=True,
        linkage_method=LinkageMethod.WARD,
        dendrogram_plot=dendro_gen,
    )
    assert medoids["C1"].pop() in ["1dq8.CS.0", "1dq9.CS.0"]
    assert medoids["C1"].pop() in ["1dq8.CS.0", "1dq9.CS.0"]
    assert len(medoids["C1"]) == 0
    assert images_identical(dendro_ref, dendro_gen)


def test_calc_univariate_hca():
    pm.reinitialize()
    
    load_ftmap(
        filename=f"{pkg_data}/1dq8_atlas.pdb",
        group="1dq8",
        deep_search=True,
        remove_nested=False
    )
    load_ftmap(
        filename=f"{pkg_data}/1dq9_atlas.pdb",
        group="1dq9",
        deep_search=True,
        remove_nested=False
    )

    dendro_ref = f"{pkg_data}/test_calc_univariate_hca_dendro_ref.svg"
    dendro_gen = f"{pkg_data}/test_calc_univariate_hca_dendro_gen.svg"
    heat_ref = f"{pkg_data}/test_calc_univariate_hca_heat_ref.svg"
    heat_gen = f"{pkg_data}/test_calc_univariate_hca_heat_gen.svg"

    calc_univariate_hca(
        sele="*.DL.*",
        overlap_function=HcaOverlapFunction.FO_AVG,
        linkage_method=LinkageMethod.COMPLETE,
        only_medoids=True,
        radius=4,
        annotate=False,
        nclusters=8,
        dendrogram_plot=dendro_gen,
        heatmap_plot=heat_gen,
    )
    assert images_identical(dendro_ref, dendro_gen)
    assert images_identical(heat_ref, heat_gen)


def test_fo_and_dce():
    pm.reinitialize()
    pm.fetch('1OD')
    pm.fetch('NH2')
    assert get_fo("%NH2", "%1OD", radius=3.0) == 1.0
    assert round(get_dce("%NH2", "%1OD", radius=3.0), 2) == 7.67


def test_calc_fingerprint():
    raise NotImplementedError

    pm.reinitialize()

    load_ftmap(f"{pkg_data}/1dq8_atlas.pdb", "1dq8")
    load_ftmap(f"{pkg_data}/1dq9_atlas.pdb", "1dq9")
    load_ftmap(f"{pkg_data}/1dqa_atlas.pdb", "1dqa")

    # img_gen = f"{pkg_data}/test_fpt_gen.svg"
    # img_ref = f"{pkg_data}/test_fpt_ref.svg"
    # calc_fingerprints(
    #     "1dqa.CS.0 / 1dqa.CS.1",
    #     site="1dqa.CS.0 | 1dqa.CS.1",
    #     site_radius=4,
    #     sharex=True,
    #     fingerprints_axis=img_gen,
    #     nbins=50,
    # )
    # assert images_identical(img_ref, img_gen)

    img_gen1 = f"{pkg_data}/test_fpt1_gen.svg"
    img_gen2 = f"{pkg_data}/test_fpt2_gen.svg"
    img_ref1 = f"{pkg_data}/test_fpt1_ref.svg"
    img_ref2 = f"{pkg_data}/test_fpt2_ref.svg"

    calc_fingerprints(
        multi_seles="1dq8.D* | 1dq8.B* / 1dq9.DL.0 / 1dqa.CS.0",
        site_radius=4.0,
        nbins=50,
        sharex=False,
        share_ylim=False,
        fingerprints_axis=img_gen1,
        dendrogram_axis=img_gen2,
    )
    assert images_identical(img_ref1, img_gen1)
    assert images_identical(img_ref2, img_gen2)


def test_res_sim():
    pm.reinitialize()
    ftmap8 = load_ftmap(f"{pkg_data}/1dq8_atlas.pdb", "1dq8")
    ftmap9 = load_ftmap(f"{pkg_data}/1dq9_atlas.pdb", "1dq9")
    assert res_sim(
        '1dq8.DL.0',
        '1dq9.CS.0',
        method=HcaOverlapFunction.JACCARD,
        radius=4.0
    ) - 0.272 < 0.01
    assert res_sim(
        '1dq8.DL.0',
        '1dq9.CS.0',
        method=HcaOverlapFunction.OVERLAP,
        radius=4.0
    ) == 0.9375


def test_load():
    pm.reinitialize()
    
    ftmap = load_ftmap(f"{pkg_data}/2TPR.pdb", deep_search=True)
    hotspots = ftmap.hotspots
    assert len(hotspots) == 42
    assert hotspots[0].Class == 'DL'

    ftmap = load_ftmap(
        f"{pkg_data}/1dqa_atlas.pdb",
        "1dqa",
        clash_threshold=0.15,
        deep_search=True
    )
    assert len(ftmap.hotspots) == 307

