import os.path
import sys
from pymol import cmd as pm
import numpy as np
import matplotlib as mpl
from xdrugpy.hotspots import (
    load_ftmap,
    plot_euclidean_hca,
    plot_pairwise_hca,
    PairwiseFunction,
    fpt_sim,
    res_sim,
    get_ho,
    get_fo,
    get_dce,
    ResidueSimilarityMethod,
    show_hs,
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
    ftmap = load_ftmap([
            f"{pkg_data}/1dq8_atlas.pdb",
            f"{pkg_data}/1dq9_atlas.pdb",
        ],
        groups=['1dq8', '1dq9'],
    )
    expr = "*.D.*"
    img_ref = f"{pkg_data}/test_euclidean_hca_ref.svg"
    img_gen = f"{pkg_data}/test_euclidean_hca_gen.svg"
    dendro, medoids = plot_euclidean_hca(
        expr,
        color_threshold=4,
        annotate=True,
        linkage_method='ward',
        plot=img_gen
    )
    assert medoids["C1"].pop() in ["1dq9.D.00", "1dq8.D.00"]
    assert medoids["C1"].pop() in ["1dq9.D.00", "1dq8.D.00"]
    assert len(medoids["C1"]) == 0
    assert images_identical(img_ref, img_gen)


def test_pairwise_hca():
    pm.reinitialize()
    ftmap = load_ftmap([
            f"{pkg_data}/1dq8_atlas.pdb",
            f"{pkg_data}/1dq9_atlas.pdb",
        ],
        groups=['1dq8', '1dq9'],
    )
    expr = "*.DS.* *.BS.*"
    
    img_ref = f"{pkg_data}/test_pairwise_hca_ref.svg"
    img_gen = f"{pkg_data}/test_pairwise_hca_gen.svg"
    plot_pairwise_hca(
        expr,
        function=PairwiseFunction.RESIDUE_JACCARD,
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
    assert round(get_ho('1dq8.DS.00', '1dq9.D.00'), 3) == 0.838


def test_fo_and_dce():
    pm.reinitialize()
    pm.fetch('1OD')
    pm.fetch('NH2')
    assert get_fo("%NH2", "%1OD", radius=3.0) == 1.0
    assert round(get_dce("%NH2", "%1OD", radius=3.0), 2) == 7.67


def test_fpt():
    pm.reinitialize()

    load_ftmap(f"{pkg_data}/2JK6.pdb")
    load_ftmap(f"{pkg_data}/2W0H.pdb")
    load_ftmap(f"{pkg_data}/6BU7.pdb")

    img_gen = f"{pkg_data}/test_fpt_gen.svg"
    img_ref = f"{pkg_data}/test_fpt_ref.svg"
    fpt_sim(
        "2JK6.CS.000_* / 2W0H.CS.001_*",
        site="2JK6.CS.000_* | 2W0H.CS.001_*",
        site_radius=5,
        contact_radius=10,
        plot_fingerprints=img_gen,
        nbins=15,
    )
    assert images_identical(img_ref, img_gen)

    img_gen1 = f"{pkg_data}/test_fpt1_gen.svg"
    img_gen2 = f"{pkg_data}/test_fpt2_gen.svg"
    img_ref1 = f"{pkg_data}/test_fpt1_ref.svg"
    img_ref2 = f"{pkg_data}/test_fpt2_ref.svg"
    fpt_sim(
        "2JK6.DL.00 / 2W0H.BL.00 / 6BU7.CS.*",
        site_radius=4.0,
        nbins=50,
        sharex=True,
        share_ylim=False,
        plot_fingerprints=img_gen1,
        plot_hca=img_gen2,
        seq_align_omega=True,
    )
    assert images_identical(img_ref1, img_gen1)
    assert images_identical(img_ref2, img_gen2)

    img_gen3 = f"{pkg_data}/test_fpt3_gen.svg"
    img_ref3 = f"{pkg_data}/test_fpt3_ref.svg"
    fpt_sim(
        "2JK6.DL.00 / 2W0H.BL.00",
        site="2JK6.DL.00",
        nbins=50,
        sharex=False,
        share_ylim=True,
        plot_fingerprints=img_gen3,
        seq_align_omega=False,
    )
    assert images_identical(img_ref3, img_gen3)


def test_res_sim():
    pm.reinitialize()
    load_ftmap(f"{pkg_data}/1dq8_atlas.pdb", "1dq8")
    load_ftmap(f"{pkg_data}/1dq9_atlas.pdb", "1dq9")
    assert round(res_sim(
        '1dq8.DS.00',
        '1dq9.DS.00',
        method=ResidueSimilarityMethod.JACCARD,
        radius=3.0
    ), 3) == 0.769
    assert res_sim(
        '1dq8.DS.00',
        '1dq9.DS.00',
        method=ResidueSimilarityMethod.OVERLAP,
        radius=4.0
    ) == 1.0


## Very ugly code ahead
##
# def test_bekar_cesaretli_2025():
#     pm.reinitialize()
#     ftmap = load_ftmap([
#         pkg_data + "/3mer_c16.pdb",
#         pkg_data + "/3mer_c16-2.pdb"
#     ], groups=["3mer_c16", "3mer_c16-2"],
#         bekar_label='MyObject'
#     )
#     assert ftmap.bekar25
#     assert ftmap.k15d_count == 2
#     assert ftmap.cs16_count == 2
#     assert 'BC25' == pm.get_property('Type', '_BC25_MyObject')
#     assert pm.get_property('IsBekar', '_BC25_MyObject') == True

#     pm.reinitialize()
#     ftmap = load_ftmap([
#         pkg_data + "/3mer_c10.pdb",
#         pkg_data + "/3mer_c16.pdb"
#     ], groups=["3mer_c10", "3mer_c16"],
#         bekar_label='MyObject'
#     )
#     assert not ftmap.bekar25
#     assert ftmap.k15d_count == 1
#     assert ftmap.cs16_count == 2
#     assert '_BC25_MyObject' in pm.get_object_list()
#     assert pm.get_property('IsBekar', '_BC25_MyObject') == False


def test_load():
    # FIXME tudo errado?
    pm.reinitialize()
    ftmap = load_ftmap([
        pkg_data + "/2TPR.pdb",
        pkg_data + "/1dq8_atlas.pdb",
        pkg_data + "/1BZL.pdb",
    ])
    hotspots = ftmap[0].hotspots
    assert len(hotspots) == 5
    assert hotspots[4].klass == 'DS' 

    hotspots = ftmap[1].hotspots
    assert len(hotspots) == 5
    assert hotspots[1].klass == 'D'
    assert hotspots[2].klass == 'DS'
    assert hotspots[4].klass == 'B'

    hotspots = ftmap[2].hotspots
    assert len(hotspots) == 3

def test_show_hs():
    pm.reinitialize()
    ftmap = load_ftmap(f"{pkg_data}/1BZL_atlas.pdb", "1BZL")

    hs = show_hs(['*.CS.000_*', '*.CS.002_*', "*.CS.004_*"])
    assert not hs.isComplex
    assert hs.nComponents == 1
