import pytest
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock
from pathlib import Path
from pymol import cmd as pm
from xdrugpy.docking import VinaDockingEngine
from xdrugpy.utils import RECEPTOR_LIBRARIES_DIR


pkg_data = Path(__file__).parent / 'data'


def test_vina_engine():
    pm.reinitialize()
    #
    # New docking
    #
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pm.load(pkg_data / '1dq8_atlas.pdb')
        eng1 = VinaDockingEngine(tmpdir, MagicMock())
        eng1.prepare_receptor(
            '%protein & polymer',
            'resi 698 and chain B',
            box_margin=3.0,
            save_lib='test_receptor'
        )
        #assert 'Starting new docking at:' in eng1.log_html.call_args_list[0][0][0]
        assert 'Adding receptor hydrogens:' in eng1.manager.logEvent.emit.call_args_list[0][0][0]
        assert 'Preparing receptor.' in eng1.manager.logEvent.emit.call_args_list[1][0][0]
        assert 'Stored receptor at:' in eng1.manager.logEvent.emit.call_args_list[3][0][0]
        with open(tmpdir / 'receptor.pdbqt') as f:
            assert len(f.readlines()) == 7544
        eng1.prepare_ligands(pkg_data / 'MiniFrag80.sdf', save_lib='minifrags')
        ligands = list((eng1.project_dir / 'queue').iterdir())
        assert "Scrubbing ligands." in eng1.manager.logEvent.emit.call_args_list[4][0][0]
        scrub_log = eng1.manager.logEvent.emit.call_args_list[5][0][0]
        assert "nr conformers:  5" in scrub_log or "nr conformers:  4 " in scrub_log # depending on scrub BUG
        assert len(ligands) in [4, 5] # depending on scrub BUG
        with open(tmpdir / 'queue' / 'Z1184909877.pdbqt') as f:
            assert len(f.readlines()) == 16
        assert "Converting ligands to PDBQT." in eng1.manager.logEvent.emit.call_args_list[6][0][0]
        lib_dir = RECEPTOR_LIBRARIES_DIR / 'minifrags'
        assert len(list(lib_dir.iterdir())) in [4, 5] # depending on scrub BUG
    #
    # Restoring libraries
    #
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        eng2 = VinaDockingEngine(tmpdir, MagicMock())
        eng2.prepare_receptor(from_lib="test_receptor")
        eng2.prepare_ligands(from_lib="minifrags")
        with open(tmpdir / 'receptor.pdbqt') as f:
            assert len(f.readlines()) == 7544
        with open(tmpdir / 'queue' / 'Z1184909877.pdbqt') as f:
            assert len(f.readlines()) == 16
        eng2.run_docking()
        assert "--cpu 1 --seed 42 --size_x 11.11 --size_y 10.48 --size_z 12.50 --center_x 16.55 --center_y -14.26 --center_z 8.36" in eng2.manager.logEvent.emit.call_args_list[2][0][0]
        assert "<b>Results:</b> 5" in eng2.manager.logEvent.emit.call_args_list[4][0][0] or "<b>Results:</b> 4" in eng2.manager.logEvent.emit.call_args_list[4][0][0] # depending on scrub BUG
        with open(tmpdir / 'results' / 'Z1184909877_out.pdbqt') as f:
            lines = f.readlines()
            assert lines[0] == 'MODEL 1\n'
            remark_row = lines[1].split()
            assert remark_row[:3] == ['REMARK', 'VINA', 'RESULT:']
            assert -3 < float(remark_row[3]) < -2
        #
        # Continuating a previous docking
        #
        eng3 = VinaDockingEngine(tmpdir, MagicMock())
        eng3.run_docking(continuation=True)
        assert "Continuating docking at:" in eng3.manager.logEvent.emit.call_args_list[1][0][0]
        assert "<b>Results:</b> 5" in eng3.manager.logEvent.emit.call_args_list[4][0][0] or "<b>Results:</b> 4" in eng3.manager.logEvent.emit.call_args_list[4][0][0] # depending on scrub BUG
