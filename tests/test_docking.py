from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call
from pathlib import Path
from pymol import cmd as pm
from xdrugpy.docking import VinaEngine
from xdrugpy import RECEPTOR_LIBRARIES_DIR, LIGAND_LIBRARIES_DIR


pkg_data = Path(__file__).parent / "data"


def test_vina_engine():
    pm.reinitialize()
    #
    # New docking
    #
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pm.load(pkg_data / "1dq8_atlas.pdb")
        eng1 = VinaEngine(tmpdir, MagicMock())
        eng1.log = MagicMock()
        eng1.cmd.run = MagicMock(wraps=eng1.cmd.run)

        ## PREPARE RECEPTOR
        eng1.prepare_receptor(
            "%protein & polymer",
            "resi 698 and chain B",
            box_margin=5.0,
            save_lib="test_receptor",
        )
        eng1.cmd.run.assert_has_calls([
            call(
                'ADDING_RECEPTOR_HYDROGENS',
                f'pdb2pqr --keep-chain --whitespace --ff PARSE --pdb-output "{tmpdir}/receptor.pdb" --with-ph 7.0 "{tmpdir}/receptor.pdb" "{tmpdir}/receptor.pqr"'
            ),
            call(
                'PREPARING_RECEPTOR',
                f'python -m meeko.cli.mk_prepare_receptor  --read_pdb "{tmpdir}/receptor.pdb" -p "{tmpdir}/receptor.pdbqt" --default_altloc A --box_center 16.55 -14.26 8.36 --box_size 15.11 14.48 16.50'
            )
        ])
        assert 602720 == len((tmpdir / "receptor.pdbqt").read_text())

        ## PREPARE LIGANDS
        eng1.prepare_ligands([str(pkg_data / "MiniFrag80.sdf")], save_lib="minifrags")
        ligands = list((eng1.project_dir / "queue").iterdir())
        assert len(ligands) in [4, 5] # depending on scrub.py bug
        assert 792 == len((tmpdir / "queue" / "Z1184909877.pdbqt").read_text())
        
    #
    # Restoring libraries and running
    #
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        eng2 = VinaEngine(tmpdir, MagicMock())
        eng2.log = MagicMock()
        eng2.prepare_receptor(from_lib="test_receptor")
        eng2.prepare_ligands(from_lib="minifrags")
        
        assert 602720 == len((tmpdir / "receptor.pdbqt").read_text())

        ligands = list((eng2.project_dir / "queue").iterdir())
        assert len(ligands) in [4, 5] # depending on scrub.py bug
        assert 792 == len((tmpdir / "queue" / "Z1184909877.pdbqt").read_text())

        eng2.run_docking()
        vina_command = (tmpdir / "vina_args.txt").read_text().strip()
        assert vina_command == (
            f'vina --verbose 0 --scoring vinardo --cpu 1 --seed 42 --size_x 15.11 --size_y 14.48 --size_z 16.50'
            f' --center_x 16.55 --center_y -14.26 --center_z 8.36 --exhaustiveness 8 --num_modes 9 --min_rmsd 1.0 --energy_range 3.0'
            f' --receptor "{tmpdir}/receptor.pdbqt" --dir "{tmpdir}/results" --batch "{tmpdir}/queue"'
        )
            