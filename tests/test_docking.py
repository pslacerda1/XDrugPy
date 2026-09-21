from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call
from pathlib import Path
from pymol import cmd as pm
from xdrugpy.docking import VinaEngine


pkg_data = Path(__file__).parent / "data"


def test_vina_engine():
    pm.reinitialize()
    #
    # New docking
    #
    with TemporaryDirectory(prefix='XDrugPy-test-') as tmpdir:
        tmpdir = Path(tmpdir)
        pm.load(str(pkg_data / "1dq8_atlas.pdb"))
        eng1 = VinaEngine(tmpdir, MagicMock())
        eng1.cmd.run = MagicMock(wraps=eng1.cmd.run)

        eng1.prepare_receptor(
            "%protein & polymer",
            "resi 698 and chain B",
            box_margin=5.0,
            save_lib="test_receptor",
        )
        assert eng1.cmd.run.call_count == 2
        assert eng1.cmd.run.call_args_list[0] == call(
            'ADDING_RECEPTOR_HYDROGENS',
            f'pdb2pqr --keep-chain --whitespace --ff PARSE --pdb-output "{tmpdir}/receptor2.pdb" --with-ph 7.0 "{tmpdir}/receptor1.pdb" "{tmpdir}/receptor.pqr"',
        )
        assert eng1.cmd.run.call_args_list[1] == call(
            'PREPARING_RECEPTOR',
            f'python -m meeko.cli.mk_prepare_receptor  --read_pdb "{tmpdir}/receptor2.pdb" -p "{tmpdir}/receptor.pdbqt" --default_altloc A --box_center 16.55 -14.26 8.36 --box_size 15.11 14.48 16.50'
        )
        assert 602720 == len((tmpdir / "receptor.pdbqt").read_text())

        ## PREPARE LIGANDS
        eng1.prepare_ligands([str(pkg_data / "MiniFrag80.sdf")], save_lib="minifrags")

        assert eng1.cmd.run.call_args_list[2] == call(
            'SCRUBBING_LIGANDS',
            f'python "/home/peu/miniconda3/envs/PyMOL/bin/scrub.py" -o "{tmpdir}/ligands_0.sdf" --cpu=1 --etkdg_rng_seed=0 --ph_high=7.0 --ph_low=7.0 --skip_acidbase --skip_tautomers "/home/peu/Desktop/XDrugPy/tests/data/MiniFrag80.sdf"'
        )
        assert eng1.cmd.run.call_args_list[3] == call(
            'CONVERTING_LIGANDS_TO_PDBQT',
            f'python -m meeko.cli.mk_prepare_ligand -i "{tmpdir}/ligands_0.sdf" --multimol_outdir "{tmpdir}/queue"'
        )


        ligands = list((eng1.project_dir / "queue").iterdir())
        assert len(ligands) in [4, 5] # depending on scrub.py bug
        assert 792 == len((tmpdir / "queue" / "Z1184909877.pdbqt").read_text())


    #
    # Restoring libraries and running
    #
    with TemporaryDirectory(prefix='XDrugPy-test-') as tmpdir:
        tmpdir = Path(tmpdir)
        eng2 = VinaEngine(tmpdir, MagicMock())
        eng2.cmd.run = MagicMock(wraps=eng2.cmd.run)
        eng2.prepare_receptor(from_lib="test_receptor")
        eng2.prepare_ligands(from_lib="minifrags")

        assert 602720 == len((tmpdir / "receptor.pdbqt").read_text())

        ligands = list((eng2.project_dir / "queue").iterdir())
        assert len(ligands) in [4, 5] # depending on scrub.py bug
        assert 792 == len((tmpdir / "queue" / "Z1184909877.pdbqt").read_text())

        eng2.run_docking()
        vina_command = (tmpdir / "vina_args.txt").read_text().strip()
        assert vina_command == (
            f'vina --verbosity 0 --scoring vinardo --cpu 1 --seed 42 --size_x 15.11 --size_y 14.48 --size_z 16.50'
            f' --center_x 16.55 --center_y -14.26 --center_z 8.36 --exhaustiveness 8 --num_modes 9 --min_rmsd 1.0 --energy_range 3.0'
            f' --receptor "{tmpdir}/receptor.pdbqt" --dir "{tmpdir}/results" --batch "{tmpdir}/queue"'
        )
        assert len(list((tmpdir / 'results').glob('*.pdbqt'))) in [4, 5] # depending on scrub.py bug
