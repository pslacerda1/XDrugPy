import os.path
from pymol import cmd as pm
import numpy as np
import matplotlib as mpl
from xdrugpy.hotspots import (
    load_ftmap,
    eftmap_overlap,
    _eftmap_overlap_get_aromatic,
    plot_euclidean_hca,
    plot_pairwise_hca,
    HeatmapFunction,
    fpt_sim,
    res_sim,
    ho,
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


def test_eftmap_overlap():
    pm.reinitialize()
    load_ftmap(f"{pkg_data}/p38_1R39.pdb")
    deloc_xyz = _eftmap_overlap_get_aromatic("ligand")
    assert deloc_xyz.shape == (17, 3)

    deloc_contacts = eftmap_overlap("ligand", "p38_1R39.ACS_aromatic_*")
    assert deloc_contacts == 12


def test_euclidean_hca():
    pm.reinitialize()
    load_ftmap(f"{pkg_data}/A7YT55_6css_atlas.pdb", "group")

    expr = "*.K15_* & p.S0>5"
    img_ref = f"{pkg_data}/test_euclidean_hca_ref.svg"
    img_gen = f"{pkg_data}/test_euclidean_hca_gen.svg"

    dendro, medoids = plot_euclidean_hca(expr, color_threshold=0.15, annotate=True, axis=img_gen)
    
    assert medoids["C1"].pop() in ["group.K15_D_00", "group.K15_D_01"]
    assert medoids["C1"].pop() in ["group.K15_D_00", "group.K15_D_01"]
    assert len(medoids["C1"]) == 0

    assert medoids["C2"].pop() in ["group.K15_D_02", "group.K15_D_03"]
    assert medoids["C2"].pop() in ["group.K15_D_02", "group.K15_D_03"]
    assert len(medoids["C2"]) == 0

    assert medoids["C3"].pop() in ["group.K15_B_01"]
    assert len(medoids["C3"]) == 0

    assert images_identical(img_ref, img_gen)


def test_pairwise_hca():
    pm.reinitialize()
    load_ftmap(f"{pkg_data}/A7YT55_6css_atlas.pdb", "A7YT55_6css")
    expr = "*.K15_*"
    img_ref = f"{pkg_data}/test_pairwise_hca_ref.svg"
    img_gen = f"{pkg_data}/test_pairwise_hca_gen.svg"
    plot_pairwise_hca(
        expr,
        method=HeatmapFunction.RESIDUE_JACCARD,
        radius=4,
        annotate=True,
        axis=img_gen,
        color_threshold=0.5
    )
    assert images_identical(img_ref, img_gen)


def test_ho():
    pm.reinitialize()
    load_ftmap([
            f"{pkg_data}/1dq8_atlas.pdb",
            f"{pkg_data}/1dq9_atlas.pdb",
        ],
        groups=['1dq8', '1dq9'],
        run_fpocket=True
    )
    assert ho('1dq9.fpocket_01', '1dq9.K15_D_00') == 0.0
    assert ho('1dq9.K15_D_00', '1dq8.K15_D_00') == 1.0


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
        "1dq8.K15_* p.ST>12 / 1dq9.K15_D_00 / 1dqa.K15_B_00",
        site="* within 4 of *_D_00",
        nbins=50,
        radius=4.0,
        axis_fingerprint=img_gen1,
        axis_dendrogram=img_gen2,
    )

    assert images_identical(img_ref1, img_gen1)
    assert images_identical(img_ref2, img_gen2)


def test_res_sim():
    pm.reinitialize()
    load_ftmap(pkg_data + "/3mer_c10.pdb")
    load_ftmap(pkg_data + "/3mer_c16.pdb")
    assert res_sim("3mer_c10.K15_B_00", "3mer_c16.K15_D_00", radius=4) == 11 / 18


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
    assert 'BC25' == pm.get_property('Type', '_bekar25_MyObject')

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
    assert '_bekar25_MyObject' not in pm.get_object_list()
