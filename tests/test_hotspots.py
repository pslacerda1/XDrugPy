import os.path
import sys
from pymol import cmd as pm
import numpy as np
import matplotlib as mpl
from xdrugpy.hotspots import (
    load_ftmap,
    plot_euclidean_hca,
    plot_pairwise_hca,
    SimilarityFunc,
    fpt_sim,
    res_sim,
    get_ho,
    get_fo,
    get_dce,
    ResidueSimilarityMethod
)
from xdrugpy.utils import RESOURCES_DIR

sys.path.append(RESOURCES_DIR)
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


def test_euclidean_hca():
    pm.reinitialize()
    load_ftmap(f"{pkg_data}/A7YT55_6css_atlas.pdb", "group", allow_nested=True)

    expr = "*.K15_*"
    img_ref = f"{pkg_data}/test_euclidean_hca_ref.svg"
    img_gen = f"{pkg_data}/test_euclidean_hca_gen.svg"

    dendro, medoids = plot_euclidean_hca(
        expr,
        color_threshold=0.15,
        annotate=True,
        plot=img_gen
    )
    
    assert medoids["C1"].pop() in ["group.K15_D_00", "group.K15_DL_00"]
    assert medoids["C1"].pop() in ["group.K15_D_00", "group.K15_DL_00"]
    assert len(medoids["C1"]) == 0

    assert medoids["C2"].pop() in ["group.K15_B_00", "group.K15_BS_00"]
    assert medoids["C2"].pop() in ["group.K15_B_00", "group.K15_BS_00"]
    assert len(medoids["C2"]) == 0

    assert images_identical(img_ref, img_gen)


def test_pairwise_hca():
    pm.reinitialize()
    ftmap = load_ftmap(f"{pkg_data}/A7YT55_6css_atlas.pdb", "A7YT55_6css", allow_nested=True)

    expr = "*.K15_*"
    img_ref = f"{pkg_data}/test_pairwise_hca_ref.svg"
    img_gen = f"{pkg_data}/test_pairwise_hca_gen.svg"

    plot_pairwise_hca(
        expr,
        function=SimilarityFunc.RESIDUE_JACCARD,
        radius=4,
        annotate=True,
        plot=img_gen,
        color_threshold=0.3
    )
    assert images_identical(img_ref, img_gen)


def test_ho():
    pm.reinitialize()
    ftmap = load_ftmap([
            f"{pkg_data}/1dq8_atlas.pdb",
            f"{pkg_data}/1dq9_atlas.pdb",
        ],
        groups=['1dq8', '1dq9']
    )
    assert get_ho('1dq8.K15_D_00', '1dq9.K15_DS_00') == 0.75


def test_fo_and_dce():
    pm.reinitialize()
    pm.fetch('1OD')
    pm.fetch('NH2')
    assert get_fo("%NH2", "%1OD", radius=3.0) == 1.0
    assert round(get_dce("%NH2", "%1OD", radius=3.0), 2) == 7.67


def test_fpt():
    pm.reinitialize()

    load_ftmap(f"{pkg_data}/1dq8_atlas.pdb", "1dq8")
    load_ftmap(f"{pkg_data}/1dq9_atlas.pdb", "1dq9")
    load_ftmap(f"{pkg_data}/1dqa_atlas.pdb", "1dqa")

    img_gen1 = f"{pkg_data}/test_fpt1_gen.svg"
    img_gen2 = f"{pkg_data}/test_fpt2_gen.svg"

    img_ref1 = f"{pkg_data}/test_fpt1_ref.svg"
    img_ref2 = f"{pkg_data}/test_fpt2_ref.svg"

    fpt_sim(
        "1dq8.K15_* / 1dq9.K15_D_00 / 1dqa.K15_B_00",
        site="*.K15_D_00",
        nbins=50,
        sharex=True,
        site_radius=4.0,
        plot_fingerprints=img_gen1,
        plot_hca=img_gen2,
    )

    assert images_identical(img_ref1, img_gen1)
    assert images_identical(img_ref2, img_gen2)


def test_res_sim():
    pm.reinitialize()
    load_ftmap(f"{pkg_data}/1dq8_atlas.pdb", "1dq8")
    load_ftmap(f"{pkg_data}/1dq9_atlas.pdb", "1dq9")
    assert res_sim(
        '1dq8.K15_D_00',
        '1dq9.K15_DS_00',
        method=ResidueSimilarityMethod.JACCARD,
        radius=4.0
    ) == 0.75
    assert res_sim(
        '1dq8.K15_D_00',
        '1dq9.K15_DS_00',
        method=ResidueSimilarityMethod.OVERLAP,
        radius=4.0
    ) == 1.0


def test_bekar_cesaretli_2025():
    pm.reinitialize()
    ftmap = load_ftmap([
        pkg_data + "/3mer_c16.pdb",
        pkg_data + "/3mer_c16-2.pdb"
    ], groups=["3mer_c16", "3mer_c16-2"],
        bekar_label='MyObject'
    )
    assert ftmap.bekar25
    assert ftmap.k15d_count == 2
    assert ftmap.cs16_count == 2
    assert 'BC25' == pm.get_property('Type', '_BC25_MyObject')
    assert pm.get_property('IsBekar', '_BC25_MyObject') == True

    pm.reinitialize()
    ftmap = load_ftmap([
        pkg_data + "/3mer_c10.pdb",
        pkg_data + "/3mer_c16.pdb"
    ], groups=["3mer_c10", "3mer_c16"],
        bekar_label='MyObject'
    )
    assert not ftmap.bekar25
    assert ftmap.k15d_count == 1
    assert ftmap.cs16_count == 2
    assert '_BC25_MyObject' in pm.get_object_list()
    assert pm.get_property('IsBekar', '_BC25_MyObject') == False


def test_load():
    pm.reinitialize()
    ftmap = load_ftmap([
        pkg_data + "/1E1V.pdb",
        pkg_data + "/1CKP.pdb",
        pkg_data + "/1K3A.pdb",
    ])
    assert len(ftmap.results[0].kozakov2015) == 1
    assert len(ftmap.results[1].kozakov2015) == 1
    assert len(ftmap.results[2].kozakov2015) == 2
    assert ftmap.results[2].kozakov2015[0].klass == 'D'
    assert ftmap.results[2].kozakov2015[1].klass == 'DL'
