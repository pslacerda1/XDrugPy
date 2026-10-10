from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call

from pymol import cmd as pm

from xdrugpy.docking import VinaEngine, parse_out_pdbqt

pkg_data = Path(__file__).parent / "data"


def test_vina_engine():

    #
    # New docking
    #
    with TemporaryDirectory(prefix='XDrugPy-test-') as tmpdir:
        tmpdir = Path(tmpdir)
        pm.load(str(pkg_data / "1dq8_atlas.pdb"))
        eng1 = VinaEngine(tmpdir, MagicMock())
        eng1.cmd.run = MagicMock(wraps=eng1.cmd.run)

        #
        # Receptor preparation
        eng1.prepare_receptor(
            "%protein & polymer",
            "resi 698 and chain B",
            box_margin=5.0,
            save_lib="test_receptor",
        )
        assert eng1.cmd.run.call_count == 2
        assert eng1.cmd.run.call_args_list[0] == call(
            'ADDING_RECEPTOR_HYDROGENS',
            f'python -m pdb2pqr --keep-chain --ff AMBER --with-ph 7.0 --whitespace "{tmpdir / "receptor.pdb" }" "{tmpdir / "receptor.pqr"}"',
        )
        assert eng1.cmd.run.call_args_list[1] == call(
            'PREPARING_RECEPTOR',
            f'python -m meeko.cli.mk_prepare_receptor --read_pdb "{tmpdir / "receptor.pdb"}" -p "{tmpdir / "receptor.pdbqt"}" --default_altloc A --box_center 16.55 -14.26 8.36 --box_size 15.11 14.48 16.50'
        )
        assert 603040 == len((tmpdir / "receptor.pdbqt").read_text())

        #
        # Ligand preparation
        eng1.prepare_ligands(
            str(pkg_data / "MiniFrag80.sdf"),
            seed=1,
            save_lib="minifrags"
        )

        assert eng1.cmd.run.call_args_list[2] == call(
            'PREPARING_LIGAND_MODELS',
            f'python -m gypsum_dl --source "{pkg_data / "MiniFrag80.sdf"}"'
            f' --output_folder "{tmpdir / "preparation"}"'
            f' --max_ph=7.0 --min_ph=7.0 --random_seed 1 --job_manager serial'
        )

        assert eng1.cmd.run.call_args_list[3] == call(
            'CONVERTING_LIGANDS_TO_PDBQT',
            f'python -m meeko.cli.mk_prepare_ligand -i "{tmpdir / "preparation" / "gypsum_dl_success.sdf" }" --multimol_outdir "{tmpdir / "queue"}"'
        )

        ligands = list((eng1.project_dir / "queue").iterdir())
        assert len(ligands) == 21
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

        assert 603040 == len((tmpdir / "receptor.pdbqt").read_text())

        ligands = list((eng2.project_dir / "queue").iterdir())
        assert len(ligands) == 21
        assert 792 == len((tmpdir / "queue" / "Z1184909877.pdbqt").read_text())

        eng2.run_docking(exhaustiveness=4)
        vina_command = (tmpdir / "vina_args.txt").read_text().strip()
        assert vina_command == (
            f'vina --verbosity 0 --scoring vinardo --cpu 1 --seed 42 --size_x 15.11 --size_y 14.48 --size_z 16.50'
            f' --center_x 16.55 --center_y -14.26 --center_z 8.36 --exhaustiveness 4 --num_modes 9 --min_rmsd 1.0 --energy_range 3.0'
            f' --receptor "{tmpdir / "receptor.pdbqt"}" --dir "{tmpdir / "results"}" --batch "{tmpdir / "queue" }"'
        )
        assert len(list((tmpdir / 'results').glob('*.pdbqt'))) == 21
        eng2.stop()
        result = parse_out_pdbqt(str(tmpdir / 'results' / 'Z1184909877-again4.pdbqt'))
        assert result[0]['name'] == 'Z1184909877-again4'
        assert -2 > result[0]['affinity'] > -3
