from pymol import cmd as pm

from xdrugpy.multi import (
    rmsf, fetch_similar, SequenceType, PROSTHETIC_GROUPS
)

from . import images_identical, ResultFigures


def test_rmsf(
    test_name,
    load_deep_1dq9,
    load_deep_1dq8,
    load_deep_2tpr,
):
    figs = ResultFigures(test_name)
    rmsf(
        selection="deep_*.protein",
        reference="deep_1dq8.protein",
        ref_site="deep_1dq8.DL.0",
        site_radius=4.0,
        axis=figs.generated,
    )
    assert images_identical(figs.generated, figs.reference)


def test_fetch_similar_0():
    pm.reinitialize()
    pm.fetch('1E92')
    data = fetch_similar(
        sequence_sele=f'1E92',
        sequence_type=SequenceType.PROTEIN,
        identity_cutoff=0.9,
        check_ligands=True,
        site_sele='1E92 and resn HBI',
        site_radius=4.0,
        ignore_ligands=PROSTHETIC_GROUPS,
        max_results=50,
    )
    assert len(pm.get_object_list()) == 17
    assert len(data) == 16
    assert ('1W0C', 1) in data
    assert ('1E7W', 1) in data
    assert len(data[('1W0C', 1)]['ligands']) == 4
    assert ('TAQ', 'A', '301') in data[('1W0C', 1)]['ligands']


def test_fetch_similar_1():
    pm.reinitialize()
    pm.fetch("1e92")
    data = fetch_similar("1E92", "protein", 0.9, max_results=20)
    assert len(pm.get_object_list()) >= 17
    assert len(data[('1W0C', 1)]['ligands']) == 0


def test_fetch_similar_organisms():
    pm.reinitialize()
    pm.fetch("1e92")
    data = fetch_similar(
        sequence_sele=f'1E92',
        identity_cutoff=0.9,
        check_ligands=True,
        site_sele='1E92 and resn HBI',
        max_results=100,
        fetch_extra=True,
    )