from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call
from pathlib import Path
from pymol import cmd as pm
from xdrugpy.docking import VinaDockingEngine
from xdrugpy.utils import RECEPTOR_LIBRARIES_DIR, LIGAND_LIBRARIES_DIR


pkg_data = Path(__file__).parent / "data"


def test_vina_engine():
    pm.reinitialize()
    #
    # New docking
    #
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pm.load(pkg_data / "1dq8_atlas.pdb")
        eng1 = VinaDockingEngine(tmpdir, MagicMock())
        eng1.log = MagicMock()
        eng1.run_cmd = MagicMock(wraps=eng1.run_cmd)
        eng1.prepare_receptor(
            "%protein & polymer",
            "resi 698 and chain B",
            box_margin=5.0,
            save_lib="test_receptor",
        )
        eng1.run_cmd.assert_has_calls([
            call(
                'ADDING_RECEPTOR_HYDROGENS',
                f'pdb2pqr --keep-chain --whitespace --ff PARSE --pdb-output "{tmpdir}/receptor.pdb" --with-ph 7.0 "{tmpdir}/receptor.pdb" "{tmpdir}/receptor.pqr"'
            ),
            call(
                'PREPARING_RECEPTOR',
                f'python -m meeko.cli.mk_prepare_receptor  --read_pdb "{tmpdir}/receptor.pdb" -p "{tmpdir}/receptor.pdbqt" --default_altloc A --box_center 16.55 -14.26 8.36 --box_size 15.11 14.48 16.50'
            )
        ])
        assert eng1.log.call_args_list[0][0][0] == 'SAVED_STORED_RECEPTOR'
        assert eng1.log.call_args_list[0][0][1]['save_lib_pdbqt'] == RECEPTOR_LIBRARIES_DIR / 'test_receptor.pdbqt'
    
        with open(tmpdir / "receptor.pdbqt") as f:
            assert len(f.readlines()) == 7534
        eng1.prepare_ligands(pkg_data / "MiniFrag80.sdf", save_lib="minifrags")
        ligands = list((eng1.project_dir / "queue").iterdir())
        assert eng1.run_cmd.call_args_list[2][0][0] == 'SCRUBBING_LIGANDS'
        assert f'--cpu=1 --etkdg_rng_seed=0 --ph=7.0 --skip_acidbase --skip_tautomers' in eng1.run_cmd.call_args_list[2][0][1]

        assert eng1.run_cmd.call_args_list[3][0][0] == 'CONVERTING_LIGANDS_TO_PDBQT'
        assert eng1.run_cmd.call_args_list[3][0][1] == f'python -m meeko.cli.mk_prepare_ligand -i "{tmpdir}/ligands.sdf" --multimol_outdir "{tmpdir}/queue"'

        assert eng1.log.call_args_list[1][0][0] == 'SAVED_STORED_LIGANDS'
        assert eng1.log.call_args_list[1][0][1]['save_lib_dir'] == LIGAND_LIBRARIES_DIR / 'minifrags'
        assert eng1.log.call_args_list[1][0][1]['n_ligands'] in [5, 4]  # depending on scrub BUG
        with open(tmpdir / "queue" / "Z1184909877.pdbqt") as f:
            assert len(f.readlines()) == 16
        
    #
    # Restoring libraries
    #
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        eng2 = VinaDockingEngine(tmpdir, MagicMock())
        eng2.log = MagicMock()
        eng2.prepare_receptor(from_lib="test_receptor")
        eng2.prepare_ligands(from_lib="minifrags")
        with open(tmpdir / "receptor.pdbqt") as f:
            assert len(f.readlines()) == 7534
        with open(tmpdir / "queue" / "Z1184909877.pdbqt") as f:
            assert len(f.readlines()) == 16
        
        eng2.run_docking()
        assert eng2.log.call_args_list[0][0][0] == 'RECOVERED_STORED_RECEPTOR'
        assert eng2.log.call_args_list[1][0][0] == 'RECOVERED_STORED_LIGANDS'
        assert eng2.log.call_args_list[2][0][0] == 'STATE_CHECKPOINTED'
        assert eng2.log.call_args_list[3][0][0] == 'RUNNING_DOCKING'
        vina_command = eng2.log.call_args_list[3][0][1]['vina_command']
        assert vina_command == (
            f'vina --receptor "{tmpdir}/receptor.pdbqt" --scoring vinardo --cpu 1 --seed 42 --size_x 15.11 --size_y 14.48 --size_z 16.50'
            f' --center_x 16.55 --center_y -14.26 --center_z 8.36 --exhaustiveness 8 --num_modes 9 --min_rmsd 1.0 --energy_range 3.0'
            f' --dir "{tmpdir}/results" --batch "{tmpdir}/queue"'
        )
        assert eng2.log.call_args_list[4][0][0] == 'DOCKING_FINISHED'
        assert eng2.log.call_args_list[5][0][0] == 'DOCKING_SUMMARY'
        assert eng2.log.call_args_list[5][0][1]['n_ligands'] in [5, 4]  # depending on scrub BUG
        assert eng2.log.call_args_list[5][0][1]['n_results'] in [5, 4]  # depending on scrub BUG
        assert eng2.log.call_args_list[5][0][1]['n_queue'] in [0, 1]  # depending on scrub BUG
        
        with open(tmpdir / "results" / "Z1184909877.pdbqt") as f:
            lines = f.readlines()
            assert lines[0] == "MODEL 1\n"
            remark_row = lines[1].split()
            assert remark_row[:3] == ["REMARK", "VINA", "RESULT:"]
            assert -3 < float(remark_row[3]) < -2
            