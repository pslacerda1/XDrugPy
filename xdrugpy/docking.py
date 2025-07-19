import os
from os.path import (
    expanduser,
    dirname,
    splitext,
    basename,
    exists
)
from time import sleep
from glob import glob
import itertools
from operator import itemgetter
import shutil
import textwrap
import subprocess
import json
from collections import Counter, OrderedDict

import pymol
import pymol.gui
from pymol import cmd as pm
from pymol.cgo import CYLINDER, SPHERE, COLOR
from pymol import Qt
import numpy as np
import pandas as pd
from lxml import etree
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean

from .utils import LIGAND_LIBRARIES_DIR, TEMPDIR, run, plot_hca_base, RECEPTOR_LIBRARIES_DIR, LIGAND_LIBRARIES_DIR


QObject = Qt.QtCore.QObject
QWidget = Qt.QtWidgets.QWidget
QFileDialog = Qt.QtWidgets.QFileDialog
QFormLayout = Qt.QtWidgets.QFormLayout
QPushButton = Qt.QtWidgets.QPushButton
QSpinBox = Qt.QtWidgets.QSpinBox
QDoubleSpinBox = Qt.QtWidgets.QDoubleSpinBox
QDockWidget = Qt.QtWidgets.QDockWidget
QLineEdit = Qt.QtWidgets.QLineEdit
QCheckBox = Qt.QtWidgets.QCheckBox
QApplication = Qt.QtWidgets.QApplication
QMessageBox = Qt.QtWidgets.QMessageBox
QVBoxLayout = Qt.QtWidgets.QVBoxLayout
QTextEdit = Qt.QtWidgets.QTextEdit
QDialog = Qt.QtWidgets.QDialog
QDialogButtonBox = Qt.QtWidgets.QDialogButtonBox
QDesktopWidget = Qt.QtWidgets.QDesktopWidget
QProgressBar = Qt.QtWidgets.QProgressBar
QAction = Qt.QtWidgets.QAction
QComboBox = Qt.QtWidgets.QComboBox
QTabWidget = Qt.QtWidgets.QTabWidget
QTableWidget = Qt.QtWidgets.QTableWidget
QTableWidgetItem = Qt.QtWidgets.QTableWidgetItem
QHeaderView = Qt.QtWidgets.QHeaderView
QFrame = Qt.QtWidgets.QFrame
QDialogButtonBox = Qt.QtWidgets.QDialogButtonBox
QTreeView = Qt.QtWidgets.QTreeView

LeftDockWidgetArea = Qt.QtCore.Qt.LeftDockWidgetArea
QRegExp = Qt.QtCore.QRegExp
QtCore = Qt.QtCore
QThread = Qt.QtCore.QThread
pyqtSignal = Qt.QtCore.Signal
QStandardPaths = Qt.QtCore.QStandardPaths

QRegExpValidator = Qt.QtGui.QRegExpValidator
QPalette = Qt.QtGui.QPalette
QTextDocument = Qt.QtGui.QTextDocument
QIntValidator = Qt.QtGui.QIntValidator
QTextCursor = Qt.QtGui.QTextCursor
QIcon = Qt.QtGui.QIcon
QStandardItem = Qt.QtGui.QStandardItem
QStandardItemModel = Qt.QtGui.QStandardItemModel


#
# General utilities
#



def display_box_sel(name, sel, margin):
    coords = pm.get_coords(sel)
    max = np.max(coords, axis=0) + margin
    min = np.min(coords, axis=0) - margin
    display_box(name, max, min)


def display_box(name, max_coords, min_coords):
    #
    # From the original AutoDock plugin
    #

    box = [
        [max_coords[0], min_coords[0]],
        [max_coords[1], min_coords[1]],
        [max_coords[2], min_coords[2]],
    ]
    cylinder_size = 0.2
    color = [1.0, 1.0, 1.0]

    view = pm.get_view()
    obj = []

    pm.delete("_box")

    # box_color
    for i in range(2):
        for k in range(2):
            for j in range(2):
                if i != 1:
                    obj.append(CYLINDER)
                    obj.extend([box[0][i], box[1][j], box[2][k]])
                    obj.extend([box[0][i + 1], box[1][j], box[2][k]])
                    obj.append(cylinder_size)
                    obj.extend(color)
                    obj.extend(color)
                    obj.append(COLOR)
                    obj.extend(color)
                    obj.append(SPHERE)
                    obj.extend([box[0][i], box[1][j], box[2][k], cylinder_size])

                if j != 1:
                    obj.append(CYLINDER)
                    obj.extend([box[0][i], box[1][j], box[2][k]])
                    obj.extend([box[0][i], box[1][j + 1], box[2][k]])
                    obj.append(cylinder_size)
                    obj.extend(color)
                    obj.extend(color)
                    obj.append(COLOR)
                    obj.extend(color)
                    obj.append(SPHERE)
                    obj.extend([box[0][i], box[1][j + 1], box[2][k], cylinder_size])
                if k != 1:
                    obj.append(CYLINDER)
                    obj.extend([box[0][i], box[1][j], box[2][k]])
                    obj.extend([box[0][i], box[1][j], box[2][k + 1]])
                    obj.append(cylinder_size)
                    obj.extend(color)
                    obj.extend(color)
                    obj.append(COLOR)
                    obj.extend(color)
                    obj.append(SPHERE)
                    obj.extend([box[0][i], box[1][j], box[2][k + 1], cylinder_size])
    axes = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    pm.load_cgo(obj, name)
    pm.set_view(view)



class OrderedCounter(Counter, OrderedDict):
    '''
    Counter that remembers the order elements are first encountered
    https://stackoverflow.com/a/23747652/199332
    '''
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)



###############################################
#          Load Result Pannel                 #
###############################################

class TreeItem(QStandardItem):
    def __init__(self, obj):
        super().__init__(str(obj))
        self.setFlags(self.flags() & ~QtCore.Qt.ItemIsEditable)

#
# Utilities for the analyze step
#

def parse_out_pdbqt(ligand_pdbqt):
    name = basename(ligand_pdbqt)
    name = name.rsplit('.', maxsplit=2)[0]    
    poses = []
    with open(ligand_pdbqt) as file:
        for line in file:
            if line.startswith("MODEL"):
                _, mode_txt = line.split()
                mode = int(mode_txt)
            elif line.startswith("REMARK VINA RESULT:"):
                parts = line.split()
                affinity = float(parts[3])
                poses.append({
                    "name": name,
                    "filename": ligand_pdbqt,
                    "affinity": affinity,
                    "mode": mode
                })
    return poses


def load_plip_pose(receptor_pdbqt, ligand_pdbqt, mode):
    plip_pdb = '%s/plip.pdb' % TEMPDIR
    plip_pse = '%s/PLIP_PROTEIN_LIG_Z_1.pse' % TEMPDIR

    pm.delete("*")
    pm.load(ligand_pdbqt, 'lig', multiplex=1, zoom=0)
    if pm.get_object_list("%lig") and  pm.count_states('%lig') == 1:
        pass
    else:
        pm.set_name(f'lig_{str(mode).zfill(4)}', 'lig')
    pm.delete('lig_*')
    pm.alter('lig', 'chain="Z"; resn="LIG"; resi=1')

    pm.load(receptor_pdbqt, 'prot')
    pm.save(plip_pdb, selection="*")
    
    command = f'python -m plip.plipcmd -qs -f "{plip_pdb}" -y -o "{TEMPDIR}"'
    output, success = run(command, cwd=TEMPDIR)
    pm.load(plip_pse)
    pm.valence('guess', 'all')


def load_plip_full(project_dir, max_load, max_mode, tree_model):
    poses = glob(f"{project_dir}/output/*.pdbqt")
    poses = itertools.chain.from_iterable(
        map(parse_out_pdbqt, poses)
    )
    poses = list(sorted(poses, key=lambda p: p['affinity']))
    pm.set('pdb_conect_all', 'off')
    pm.delete('prot')
    fname = f"{project_dir}/receptor.pdbqt"
    pm.load(fname, 'prot')
    pm.alter('prot', "type='ATOM'")
    interactions = []
    interactions_type = [
        "hydrophobic_interaction",
        "hydrogen_bond",
        "water_bridge",
        "salt_bridge",
        "pi_stack",
        "pi_cation_interaction",
        "halogen_bond",
        "metal_complex"
    ]
    count = 0
    for pose in poses:
        if pose['mode'] > max_mode:
            continue
        count += 1
        if count > max_load:
            break

        pm.delete('%lig')
        name = pose["name"]
        mode = pose["mode"]
        in_fname = project_dir + f'/output/{name}.pdbqt'
        out_fname = TEMPDIR + f'/plip.{name}-{mode}.pdb'
        pm.load(in_fname, 'lig', multiplex=1, zoom=0)
        if pm.get_object_list("%lig") and  pm.count_states('%lig') == 1:
            pass
        else:
            pm.set_name(f'lig_{str(mode).zfill(4)}', 'lig')
        pm.delete('lig_*')
        pm.alter('lig', 'chain="Z"')
        pm.alter('lig', 'resn="LIG"')
        pm.alter('lig', 'resi=1')
        pm.alter('lig', 'type="HETATM"')
        pm.save(out_fname, selection='*')
        
        command = f'python -m plip.plipcmd -qs -f "{out_fname}" -x -o "{TEMPDIR}/"'
        print(f"RUNNING COMMAND: {command}")
        subprocess.run(command, cwd=TEMPDIR, shell=True)

        with open(TEMPDIR + '/report.xml') as fp:
            plip = etree.parse(fp)
        for inter_type in interactions_type:
            restype = plip.xpath(f"//{inter_type}/restype/text()")
            resnr = map(int, plip.xpath(f"//{inter_type}/resnr/text()"))
            reschain = plip.xpath(f"//{inter_type}/reschain/text()")
            for inter in zip(restype, resnr, reschain):
                interactions.append([f'{name}_m{mode}', inter_type, *inter])
    interactions = sorted(interactions, key=lambda i: (i[4], i[3], i[1]))
    residues_l = ['%s%s%s' % (i[2], i[3], i[4]) for i in interactions]
    interactions_l = [i[1] for i in interactions]
    names_l = [i[0] for i in interactions]

    for inter_type in interactions_type.copy():
        if len([i for i in interactions if i[1] == inter_type]) == 0:
            interactions_type.remove(inter_type)

    fig, axs = plt.subplots(len(interactions_type), layout="constrained", sharex=True)
    for ax, interaction_type in zip(axs, interactions_type):
        count = {}
        for res in residues_l:
            count[res] = 0
        for res, inter_type in zip(residues_l, interactions_l):
            if inter_type == interaction_type:
                count[res] += 1
        # for res, inter_type in zip(residues_l.copy(), interactions_l.copy()):
        #     if inter_type == interaction_type:
        #         if count[res] == 0:
        #             del count[res]
        ax.bar(count.keys(), count.values())
        ax.set_title(interaction_type)
    plt.xticks(rotation=90)
    plt.show()

    df = pd.DataFrame({
        'name': names_l,
        'residue': residues_l
    })
    labels = []
    mols = []
    prev_name = None
    for cur_name, cur_residues in df.groupby('name'):
        if cur_name != prev_name:
            prev_name = cur_name
            counter = OrderedCounter(residues_l)
            for res in counter:
                counter[res] = 0
            for res in cur_residues['residue']:
                counter[res] += 1
            mols.append([counter[r] for r in residues_l])
            mol_item = TreeItem(cur_name)
            tree_model.appendRow(mol_item)
            for res, count in counter.items():
                if count > 0:
                    resn = res[:3]
                    chain = res[-1:]
                    resi = res[3:-1]
                    mol_item.appendRow([
                        TreeItem(chain),
                        TreeItem(resi),
                        TreeItem(resn),
                        TreeItem(str(count))
                    ])
            labels.append(cur_name)
    
    X = []
    for idx1, mol1 in enumerate(mols):
        for idx2, mol2 in enumerate(mols):
            if idx1 >= idx2:
                continue
            d = euclidean(mol1, mol2)
            X.append(d)
    ax.set_xlim(0)
    plot_hca_base(
        X,
        labels,
        linkage_method='ward',
        color_threshold=-1,
        axis=None
    )


###############################################
#          Load Result Pannel                 #
###############################################

class TreeItem(QStandardItem):
    def __init__(self, obj):
        super().__init__(str(obj))
        self.setFlags(self.flags() & ~QtCore.Qt.ItemIsEditable)


class ResultsWidget(QWidget):

    class ResultsTableWidget(QTableWidget):
        def __init__(self, project_dir):
            super().__init__()
            self.project_dir = project_dir
            self.props = ["Name", "Mode", "Affinity"]

            self.setSelectionBehavior(QTableWidget.SelectRows)
            self.setSelectionMode(QTableWidget.SingleSelection)
            self.setColumnCount(3)
            self.setHorizontalHeaderLabels(self.props)
            header = self.horizontalHeader()
            for idx in range(len(self.props)):
                header.setSectionResizeMode(
                    idx, QHeaderView.ResizeMode.ResizeToContents
                )

            @self.itemClicked.connect
            def itemClicked(item):
                name = self.item(item.row(), 0).text()
                mode = self.item(item.row(), 1).text()
                receptor_pdbqt = '%s/receptor.pdbqt' % self.project_dir
                ligand_pdbqt = f'{self.project_dir}/output/{name}.pdbqt'
                load_plip_pose(receptor_pdbqt, ligand_pdbqt, mode)
        
    class SortableItem(QTableWidgetItem):
        def __init__(self, obj):
            super().__init__(str(obj))
            self.setFlags(self.flags() & ~QtCore.Qt.ItemIsEditable)

        def __lt__(self, other):
            try:
                return float(self.text()) < float(other.text())
            except ValueError:
                return self.text() < other.text()

    def __init__(self, is_intensive, project_dir, max_load, max_mode):
        super().__init__()
        self.is_intensive = is_intensive
        self.project_dir = project_dir
        self.max_load = max_load
        self.max_mode = max_mode

        layout = QVBoxLayout()
        self.setLayout(layout)

        tab = QTabWidget()
        layout.addWidget(tab)

        tab1_widget = QWidget()
        tab1_layout = QVBoxLayout(tab1_widget)
        tab1_widget.setLayout(tab1_layout)
        tab.addTab(tab1_widget, "Affinity list")

        self.table_widget = self.ResultsTableWidget(project_dir)
        tab1_layout.addWidget(self.table_widget)
        
        export_btn = QPushButton(QIcon("save"), "Export Table")
        export_btn.clicked.connect(self.export)
        tab1_layout.addWidget(export_btn)

        tab2_widget = QWidget()
        tab2_layout = QVBoxLayout(tab2_widget)
        tab2_widget.setLayout(tab2_layout)
        tab.addTab(tab2_widget, "Residue tree")
        
        self.tree_model = QStandardItemModel()
        self.tree_model.setHorizontalHeaderLabels(["Molecule/Chain", "Resi", "Resn", "Count"])
        self.tree_widget = QTreeView()
        self.tree_widget.setModel(self.tree_model)
        tab2_layout.addWidget(self.tree_widget) 
        
    def showEvent(self, event):
        self.refresh()
        super().showEvent(event)

    def refresh(self):
        self.table_widget.setSortingEnabled(False)

        # remove old rows
        while self.table_widget.rowCount() > 0:
            self.table_widget.removeRow(0)

        # append new rows
        project_dir = expanduser(self.project_dir)
        results = itertools.chain.from_iterable(
            map(parse_out_pdbqt, glob(f"{project_dir}/output/*.pdbqt"))
        )
        results = sorted(results, key=itemgetter("affinity"))
        count = 0 
        for idx, pose in enumerate(results):
            if pose['mode'] <= self.max_mode:
                self.appendRow(pose)
                count += 1
            if count >= self.max_load:
                break
        self.table_widget.setSortingEnabled(True)

        if self.is_intensive:
            load_plip_full(project_dir, self.max_load, self.max_mode, self.tree_model)

    def appendRow(self, pose):
        self.table_widget.insertRow(self.table_widget.rowCount())
        line = self.table_widget.rowCount() - 1

        self.table_widget.setItem(line, 0, self.SortableItem(pose['name']))
        self.table_widget.setItem(line, 1, self.SortableItem(pose['mode']))
        self.table_widget.setItem(line, 2, self.SortableItem(pose['affinity']))
        
    def export(self):
        fileDialog = QFileDialog()
        fileDialog.setNameFilter("Excel file (*.xlsx)")
        fileDialog.setViewMode(QFileDialog.Detail)
        fileDialog.setAcceptMode(QFileDialog.AcceptSave)
        fileDialog.setDefaultSuffix(".xlsx")

        if fileDialog.exec_():
            filename = fileDialog.selectedFiles()[0]
            ext = splitext(filename)[1]
            with pd.ExcelWriter(filename) as xlsx_writer:
                row_count = self.table_widget.rowCount()
                col_count = self.table_widget.columnCount()
                data = []
                for row in range(row_count):
                    row_data = []
                    for col in range(col_count):
                        item = self.table_widget.item(row, col)
                        row_data.append(item.text() if item else '')
                    data.append(row_data)
                title = basename(self.project_dir)
                df = pd.DataFrame(data, columns=['Name', 'Mode', 'Affinity'])
                df.to_excel(xlsx_writer, sheet_name=title, index=False)
                

def new_load_results_widget():
    dockWidget = QDockWidget()
    dockWidget.setWindowTitle("Analyze Vina")

    widget = QWidget()
    layout = QFormLayout(widget)
    widget.setLayout(layout)
    dockWidget.setWidget(widget)

    #
    # Max number of total loaded poses
    #
    max_load_spin = QSpinBox(widget)
    max_load_spin.setRange(1, 99999999)
    max_load_spin.setValue(15)
    max_load_spin.setGroupSeparatorShown(True)

    #
    # Only the best poses of each ligand
    #
    max_mode_spin = QSpinBox(widget)
    max_mode_spin.setRange(1, 50)
    max_mode_spin.setValue(9)
    max_mode_spin.setGroupSeparatorShown(True)

    #
    # Plot interaction histogram
    #
    intensive_check = QCheckBox()
    intensive_check.setChecked(False)

    #
    # Choose output folder
    #
    show_table_button = QPushButton("Load docking...", widget)

    @show_table_button.clicked.connect
    def load_results():
        nonlocal results_widget
        docking_file = str(
            QFileDialog.getExistingDirectory(
                show_table_button,
                "Output folder",
                expanduser("~"),
                QFileDialog.ShowDirsOnly,
            )
        )
        if not docking_file:
            return
        
        project_dir = dirname(docking_file)

        if results_widget is not None:
            results_widget.setParent(None)
        del results_widget
        results_widget = ResultsWidget(
            intensive_check.isChecked(),
            project_dir,
            max_load_spin.value(),
            max_mode_spin.value(),
        )
        layout.setWidget(5, QFormLayout.SpanningRole, results_widget)

    #
    # Results Table
    #
    results_widget = None
    
    #
    # Setup form
    #
    layout.addRow("Max load:", max_load_spin)
    layout.addRow("Max mode:", max_mode_spin)
    layout.addRow("Intensive:", intensive_check)
    layout.setWidget(4, QFormLayout.SpanningRole, show_table_button)
    widget.setLayout(layout)

    return dockWidget


###############################################
#          Run Docking Pannel                 #
###############################################

class VinaThread(QThread):
    vinaStarted = pyqtSignal()
    numSteps = pyqtSignal(int)
    setStep = pyqtSignal(int)
    logEvent = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, run_implementation, parent=None):
        super().__init__(parent)
        self.run_implementation = run_implementation

    def run(self):
        self.run_implementation(self)



class VinaThreadDialog(QDialog):

    def __init__(self, run_function, parent=None):
        super().__init__(parent)
        self.vina = VinaThread(run_function)
        self.vina.finished.connect(self._finished)
        
        # Setup window
        self.setModal(True)
        self.resize(QDesktopWidget().availableGeometry(self).size() * 0.7)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.CustomizeWindowHint)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)

        self.layout = QVBoxLayout(self)

        # Setup progress bar
        self.progress = QProgressBar()
        self.layout.addWidget(self.progress)
        self.progress.setValue(0)
        @self.vina.numSteps.connect
        def numSteps(x):
            self.progress.setMaximum(x)
        @self.vina.setStep.connect
        def setStep(x):
            self.progress.setValue(x)

        # Rich text output
        self.text = QTextEdit(self)
        self.layout.addWidget(self.text)
        self.text.setReadOnly(True)
        self.vina.logEvent.connect(self._appendHtml)

        # Ok / Cancel buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Abort, QtCore.Qt.Horizontal, self
        )
        self.layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self._ok)
        self.button_box.rejected.connect(self._abort)
        self.button_box.button(QDialogButtonBox.Ok).setDisabled(True)

        # Start docking
        self.vina.start()
        @self.vina.vinaStarted.connect
        def vinaStarted():
            # TODO start signal
            pass

    def _appendHtml(self, html):
        self.text.moveCursor(QTextCursor.End)
        self.text.insertHtml(self._prepareHtml(html))

    def _appendCodeHtml(self, html):
        self.text.moveCursor(QTextCursor.End)
        self.text.insertHtml("<pre>" + self._prepareHtml(html) + "</pre>")

    def _finished(self, status=False):
        ok_button = self.button_box.button(QDialogButtonBox.Ok)
        abort_button = self.button_box.button(QDialogButtonBox.Abort)

        ok_button.setDisabled(False)
        abort_button.setDisabled(True)

    def _ok(self):
        self.accept()
    
    def _abort(self):
        self.reject()
    
    @staticmethod
    def _prepareHtml(html):
        return textwrap.dedent(html)


#
# Run docking software
#

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional


@dataclass
class DockingEngine:
    project_dir: Path
    manager: VinaThread

    def __post_init__(self):
        self.project_dir = Path(self.project_dir)
        if self.project_dir.is_dir():
            if len([*self.project_dir.iterdir()]) > 0:
                self.log_html(f"""
                    <font color="red">
                        <b>The docking folder is not empty:</b> '{self.project_dir}'
                    </font>
                """)
        else:
            self.log_html(f"""
                <br/><b>Starting new docking at:</b> '{self.project_dir}'
            """)
            self.project_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = self.project_dir / "results"
        self.queue_dir = self.project_dir / "queue"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_receptor(
        self,
        receptor_sele: str = "",
        box_sele: str = "",
        box_margin: float = 0.0,
        allow_bad_res: bool = False,
        from_lib: str = "",
        save_lib: str = "",
    ) -> bool:
        raise NotImplementedError
    
    def prepare_ligands(
        self,
        ligands_path: str = "",
        ph: float = 7.4,
        cpu: int = 1,
        skip_acidbase = True,
        skip_tautomers = True,
        from_lib: str = "",
        save_lib: str = ""
    ) -> bool:
        raise NotImplementedError
    
    def log_html(self, message: str):
        self.manager.logEvent.emit(message)
            
    def set_num_steps(self, n_ligands: int):
        self.current_step = 0
        self.manager.numSteps.emit(n_ligands)
    
    def increment_step(self, step: int):
        self.current_step += step
        self.manager.setStep.emit(self.current_step)

    def finished(self):
        self.manager.finished.emit()

class VinaDockingEngine(DockingEngine):

    def prepare_receptor(
            self,
            receptor_sele: str = "",
            box_sele: str = "",
            box_margin: float = 0.0,
            allow_bad_res: bool = False,
            from_lib: str = "",
            save_lib: str = "",
        ) -> bool:
        self.receptor_pdbqt = self.project_dir / "receptor.pdbqt"
        from_lib_pdbqt = RECEPTOR_LIBRARIES_DIR / f"{from_lib}.pdbqt"
        save_lib_pdbqt = RECEPTOR_LIBRARIES_DIR / f"{save_lib}.pdbqt"
        from_lib_box = RECEPTOR_LIBRARIES_DIR / f"{from_lib}.box"
        save_lib_box = RECEPTOR_LIBRARIES_DIR / f"{save_lib}.box"
        if from_lib and from_lib_pdbqt.exists():
            shutil.copy(from_lib_pdbqt, self.receptor_pdbqt)
            with open(from_lib_box) as f:
                box_data = json.load(f)
            self.box_size = tuple(box_data['size'])
            self.box_center = tuple(box_data['center'])
            self.log_html(f"""
                <br/><b>Recovered stored receptor:</b> '{from_lib_pdbqt}'
            """)
            return True
        else:
            assert receptor_sele, "Receptor selection must be provided."
            assert box_sele, "Box selection must be provided."
            #
            # Get box coordinates
            #
            box_coords = pm.get_coords(box_sele)
            max = np.max(box_coords, axis=0)
            min = np.min(box_coords, axis=0)
            half_size = (max - min) / 2
            center = min + half_size
            size_x, size_y, size_z = (half_size + box_margin) * 2
            center_x, center_y, center_z = center
            self.box_size = np.array((size_x, size_y, size_z)).tolist()
            self.box_center = np.array((center_x, center_y, center_z)).tolist()
            receptor_pdb = self.project_dir / "receptor.pdb"
            self.log_html(f"""
                <br/><b>Adding receptor hydrogens:</b>
            """)
            #
            # Run Meeko to prepare the receptor
            #
            pm.h_add(receptor_sele)
            pm.save(receptor_pdb, receptor_sele)
            if allow_bad_res:
                allow_bad_res = "--allow_bad_res"
            else:
                allow_bad_res = ""
            command = (
                f'python -m meeko.cli.mk_prepare_receptor'
                f' {allow_bad_res}'
                f' --read_pdb "{receptor_pdb}"'
                f' -p "{self.receptor_pdbqt}"'
                f' --box_center {center_x:.2f} {center_y:.2f} {center_z:.2f}'
                f' --box_size {size_x:.2f} {size_y:.2f} {size_z:.2f}'
            )
            self.log_html(f"""
                <br/><b>Preparing receptor.</b>
                <br/><b>Command:</b> {command}
                <br/>
            """)
            output, success = run(command)
            self.log_html(f"""
                <pre>{output}</pre>
            """)
            if not success:
                return False
            if save_lib:
                self.log_html(f"""
                    <br/><b>Stored receptor at:</b> '{save_lib_pdbqt}'
                """)
                shutil.copy(self.receptor_pdbqt, save_lib_pdbqt)
                with open(save_lib_box, 'w') as f:
                    box_data = {
                        'size': self.box_size,
                        'center': self.box_center
                    }
                    json.dump(box_data, f, indent=4)
            return True

    def prepare_ligands(
        self,
        ligands_path: str = "",
        ph: float = 7.4,
        cpu: int = 1,
        skip_acidbase = True,
        skip_tautomers = True,
        from_lib: str = "",
        save_lib: str = ""
    ) -> bool:
        self.queue_dir = self.project_dir / "queue"
        from_lib_dir = LIGAND_LIBRARIES_DIR / from_lib
        save_lib_dir = LIGAND_LIBRARIES_DIR / save_lib
        if from_lib and from_lib_dir.exists():
            shutil.rmtree(self.queue_dir, ignore_errors=True)
            shutil.copytree(from_lib_dir, self.queue_dir)
            self.log_html(f"""
                <br/><b>Recovered stored ligands:</b> '{from_lib_dir}'
            """)
            return True
        else:
            #
            # Scrubbing ligands
            #
            ligands_file = Path(ligands_path)
            ligands_sdf = self.project_dir / "ligands.sdf"
            if skip_acidbase:
                skip_acidbase = "--skip_acidbase"
            else:
                skip_acidbase = ""
            if skip_tautomers:
                skip_tautomers = "--skip_tautomers"
            else:
                skip_tautomers = ""
            command = (
                f'python -m scrubber.main -o "{ligands_sdf}" --ph {ph} --cpu {cpu}'
                f' {skip_acidbase} {skip_tautomers} "{ligands_file}"'
            )
            self.log_html(f"""
                <br/><b>Scrubbing ligands.</b>
                <br/><b>Command:</b> {command}
            """)
            output, success = run(command)
            self.log_html(f"""
                <pre>{output}</pre>
            """)
            if not success:
                return False
            #
            # Converting to PDBQT
            #
            self.queue_dir.mkdir(parents=True, exist_ok=True)
            command = (
                f'python -m meeko.cli.mk_prepare_ligand'
                f' -i "{ligands_sdf}" --multimol_outdir "{self.queue_dir}"'
            )
            self.log_html(f"""
                <br/><b>Converting ligands to PDBQT.</b>
                <br/><b>Command:</b> {command}
                <br/>
            """)
            output, success = run(command)
            self.log_html(f"""
                <pre>{output}</pre>
            """)
            if not success:
                return False
            if save_lib:
                self.log_html(f"""
                    <br/>
                    <br/><b>Storing compound library at:</b> {save_lib_dir}
                """)
                shutil.rmtree(save_lib_dir, ignore_errors=True)
                shutil.copytree(self.queue_dir, save_lib_dir)
                return True
    
    def run_docking(
        self,
        scoring: str = "vinardo",
        exhaustiveness: int = 8,
        num_modes: int = 9,
        min_rmsd: float = 1.0,
        energy_range: float = 3.0,
        cpu: int = 1,
        seed: int = 42,
        continuation: bool = False,
    ) -> bool:
        if continuation:
            self.log_html(f"""
                <br/><b>Continuating docking at:</b> '{self.project_dir}'
            """)
            with open(self.project_dir / "vina_args.txt", 'r') as f:
                vina_command = f.readline().strip()
        else:
            vina_command = (
                "vina"
                f" --receptor '{self.receptor_pdbqt}'"
                f" --scoring {scoring}"
                f" --cpu {cpu}"
                f" --seed {seed}"
                f" --size_x {self.box_size[0]:.2f}"
                f" --size_y {self.box_size[1]:.2f}"
                f" --size_z {self.box_size[2]:.2f}"
                f" --center_x {self.box_center[0]:.2f}"
                f" --center_y {self.box_center[1]:.2f}"
                f" --center_z {self.box_center[2]:.2f}"
                f" --exhaustiveness {exhaustiveness}"
                f" --num_modes {num_modes}"
                f" --min_rmsd {min_rmsd}"
                f" --energy_range {energy_range}"
                f" --dir '{self.results_dir}'"
                f" --batch '{self.queue_dir}'"
            )
            with open(self.project_dir / "vina_args.txt", 'w') as f:
                f.write(vina_command + '\n')
        self.log_html(f"""
            <br/>
            <br/><b>Docking ligands.</b>
            <br/><b>Command:</b> {vina_command}
        """)
        #
        # Clean up the queue directory
        #
        that = self
        class WorkerJanitor(QObject):
            finished = pyqtSignal()
            def run(self):
                while True:
                    sleep(.5)
                    if len([*that.queue_dir.iterdir()]) == 0:
                        break
                    for queue_pdbqt in that.queue_dir.iterdir():
                        result_pdbqt = queue_pdbqt.name[:-6] + "_out.pdbqt"
                        result_pdbqt = that.results_dir / result_pdbqt
                        if result_pdbqt.exists():
                            queue_pdbqt.unlink()
                            that.increment_step(1)
                            break
                self.finished.emit()
        thread = QThread()
        worker = WorkerJanitor()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        def remove_threads():
            worker.deleteLater()
            thread.quit()
            thread.wait()
        worker.finished.connect(remove_threads)
        thread.start()
        #
        # Run Vina
        #
        n_results = len(glob(str(self.results_dir / '*_out.pdbqt')))
        n_queue = len(glob(str(self.queue_dir / '*.pdbqt')))
        n_ligands = n_results + n_queue
        self.set_num_steps(n_ligands)
        self.increment_step(n_results)
        output, success = run(vina_command)
        sleep(1)
        self.log_html(f"""
            <br/><b>Docking finished.</b>
        """)
        if not success:
            self.log_html(f"""
                <font color="red">
                    <br/><b>Failure on docking. Outputing last bytes.</b>
                    <br/><pre>{output[-2048:]}</pre>
                </font> 
                </br>
            """)
        n_results = len(glob(str(self.results_dir / '*_out.pdbqt')))
        n_queue = len(glob(str(self.queue_dir / '*.pdbqt')))
        n_ligands = n_results + n_queue
        self.log_html(f"""
            <hr/>
            <br/><b>Summary totals</b>
            <br/><b>Expected:</b> {n_ligands}
            <br/><b>Results:</b> {n_results}
            <br/><b>Queued:</b> {n_queue}
        """)
        remove_threads()
        self.manager.finished.emit()
        


class PyMOLComboObjectBox(QComboBox):

    def __init__(self, sele):
        super().__init__()
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.sele = sele
        self.setEditText("")

    def showPopup(self):
        currentText = self.currentText().strip()
        selections = pm.get_names("selections", enabled_only=True)
        objects = pm.get_names("objects", enabled_only=True)
        self.clear()
        self.addItems("(%s)" % s for s in selections)
        self.addItems(objects)
        if currentText != "":
            self.setCurrentText(currentText)
        super().showPopup()


def new_run_docking_widget():
    dockWidget = QDockWidget()
    dockWidget.setWindowTitle("Run Vina")

    widget = QWidget()

    layout = QFormLayout(widget)
    widget.setLayout(layout)
    dockWidget.setWidget(widget)

    ##########################################
    # RECEPTOR OPTIONS
    #
    # TAB 1

    #
    # Receptor selection
    #
    tab_receptor = QTabWidget()

    tab1_widget = QWidget()
    tab1_layout = QFormLayout(tab1_widget)
    tab1_widget.setLayout(tab1_layout)
    tab_receptor.addTab(tab1_widget, "New receptor")

    receptor_sel = PyMOLComboObjectBox("polymer")
    tab1_layout.addRow("Receptor:", receptor_sel)

    @receptor_sel.currentTextChanged.connect
    def validate(text):
        validate_receptor_sel()

    def validate_receptor_sel():
        text = receptor_sel.currentText()
        palette = QApplication.palette(receptor_sel)
        palette.setColor(QPalette.Base, QtCore.Qt.white)
        valid = True
        try:
            if pm.count_atoms(f"({text}) and polymer") == 0:
                raise
        except:
            palette.setColor(QPalette.Base, QtCore.Qt.red)
            valid = False
        receptor_sel.setPalette(palette)
        return valid

    #
    # Box selection
    #
    box_sel = PyMOLComboObjectBox("polymer")
    tab1_layout.addRow("Box:", box_sel)
    
    @box_sel.currentTextChanged.connect
    def validate(text):
        validate_box_sel()

    def validate_box_sel():
        text = box_sel.currentText()
        palette = QApplication.palette(box_sel)
        palette.setColor(QPalette.Base, QtCore.Qt.white)
        try:
            if pm.count_atoms(text) == 0:
                raise
        except:
            palette.setColor(QPalette.Base, QtCore.Qt.red)
            box_sel.setPalette(palette)
            pm.delete("box")
            return False
        display_box_sel("box", text, box_margin_spin.value())
        box_sel.setPalette(palette)
        return True
        
    box_margin_spin = QDoubleSpinBox(widget)
    box_margin_spin.setRange(0.0, 10.0)
    box_margin_spin.setValue(3.0)
    box_margin_spin.setSingleStep(0.1)
    box_margin_spin.setDecimals(1)
    tab1_layout.addRow("Box margin:", box_margin_spin)

    allow_bad_res_check = QCheckBox()
    allow_bad_res_check.setChecked(False)
    tab1_layout.addRow("Allow bad residues:", allow_bad_res_check)

    @box_margin_spin.valueChanged.connect
    def display_box(margin):
        pm.delete("box")
        display_box_sel("box", box_sel.currentText(), margin)

    saved_receptor_line = QLineEdit()
    saved_receptor_line.setPlaceholderText("set a title to save")
    tab1_layout.addRow("Store receptor:", saved_receptor_line)
    
    #
    # TAB 2
    #
    rec_tab2_widget = QWidget()
    rec_tab2_layout = QFormLayout(rec_tab2_widget)
    rec_tab2_widget.setLayout(rec_tab2_layout)
    tab_receptor.addTab(rec_tab2_widget, "Stored receptor")


    tab_rec_idx = 0
    rec_library_combo = QComboBox()
    rec_library_combo.addItems(os.listdir(RECEPTOR_LIBRARIES_DIR))
    rec_tab2_layout.addRow("Library:", rec_library_combo)
    @tab_receptor.currentChanged.connect
    def tab_changed(idx):
        nonlocal tab_rec_idx
        tab_rec_idx = idx
        if idx == 1:
            rec_library_combo.clear()
            for lib_fname in os.listdir(RECEPTOR_LIBRARIES_DIR):
                if lib_fname.endswith(".box"):
                    continue
                recetor_lib = lib_fname[:-6]
                rec_library_combo.addItem(recetor_lib)
    

    ##########################################
    # LIGAND OPTIONS
    #
    # TAB 1
    # Choose ligand files
    #
    tab_ligand = QTabWidget()
    
    tab1_widget = QWidget()
    tab1_layout = QFormLayout(tab1_widget)
    tab1_widget.setLayout(tab1_layout)
    tab_ligand.addTab(tab1_widget, "New library")

    ligands_file = None
    ligands_button = QPushButton("Choose file...", widget)
    tab1_layout.addRow("Ligands file:", ligands_button)

    ph_spin = QDoubleSpinBox(widget)
    ph_spin.setRange(0.0, 14.0)
    ph_spin.setValue(7.0)
    ph_spin.setSingleStep(0.1)
    ph_spin.setDecimals(1)
    tab1_layout.addRow("Ligands pH:", ph_spin)

    enumerate_acidbase = QCheckBox()
    enumerate_acidbase.setChecked(False)
    tab1_layout.addRow("Enumerate acid-base:", enumerate_acidbase)

    enumerate_tautomers = QCheckBox()
    enumerate_tautomers.setChecked(False)
    tab1_layout.addRow("Enumerate tautomers:", enumerate_tautomers)

    @ligands_button.clicked.connect
    def choose_ligands():
        nonlocal ligands_file
        ligands_file = str(
            QFileDialog.getOpenFileName(
                ligands_button, "Ligand files", expanduser("~"), "Molecular ligand files (*.smi *.sdf *.mol *.mol2)"
            )[0]
        )
        if not ligands_file:
            return
        ligands_button.setText(basename(ligands_file))

    #
    # Molecular library
    #
    compound_library_line = QLineEdit()
    compound_library_line.setPlaceholderText("set a title to save")
    tab1_layout.addRow("Store library:", compound_library_line)

    #
    # TAB 2
    # Choose ligand library
    #

    lig_tab2_widget = QWidget()
    lig_tab2_layout = QFormLayout(lig_tab2_widget)
    lig_tab2_widget.setLayout(lig_tab2_layout)
    tab_ligand.addTab(lig_tab2_widget, "Stored library")

    tab_lig_idx = 0
    library_combo = QComboBox()
    library_combo.addItems(os.listdir(LIGAND_LIBRARIES_DIR))
    lig_tab2_layout.addRow("Library:", library_combo)
    @tab_ligand.currentChanged.connect
    def tab_changed(idx):
        nonlocal tab_lig_idx
        tab_lig_idx = idx
        if idx == 1:
            library_combo.clear()
            library_combo.addItems(os.listdir(LIGAND_LIBRARIES_DIR))

    ##########################################

    options_group = QWidget()
    options_group_layout = QFormLayout(options_group)
    options_group.setLayout(options_group_layout)

    function = QComboBox(widget)
    function.addItems(["vina", "vinardo", "ad4"])
    options_group_layout.addRow("Function:", function)

    exhaustiveness_spin = QSpinBox(widget)
    exhaustiveness_spin.setRange(1, 50)
    exhaustiveness_spin.setValue(8)
    options_group_layout.addRow("Exhaustiveness:", exhaustiveness_spin)

    num_modes_spin = QSpinBox(widget)
    num_modes_spin.setRange(1, 100)
    num_modes_spin.setValue(3)
    options_group_layout.addRow("Number of modes", num_modes_spin)

    min_rmsd_spin = QDoubleSpinBox(widget)
    min_rmsd_spin.setRange(0.0, 3.0)
    min_rmsd_spin.setValue(1.0)
    options_group_layout.addRow("Minimum RMSD:", min_rmsd_spin)

    energy_range_spin = QDoubleSpinBox(widget)
    energy_range_spin.setRange(0, 10.0)
    energy_range_spin.setValue(3.0)
    options_group_layout.addRow("Energy range:", energy_range_spin)

    cpu_count = QThread.idealThreadCount()
    cpu_spin = QSpinBox(widget)
    cpu_spin.setRange(1, cpu_count)
    cpu_spin.setValue(cpu_count)
    options_group_layout.addRow("Number of CPUs:", cpu_spin)

    seed_spin = QSpinBox(widget)
    seed_spin.setRange(1, 10000)
    seed_spin.setValue(1)
    options_group_layout.addRow("Seed number:", seed_spin)

    #
    # Choose  output folder
    #
    continuation_check = QCheckBox()
    @continuation_check.stateChanged.connect
    def stateChanged(state):
        if state == 2:
            tab_receptor.setEnabled(False)
            tab_ligand.setEnabled(False)
            options_group.setEnabled(False)
        else:
            tab_receptor.setEnabled(True)
            tab_ligand.setEnabled(True)
            options_group.setEnabled(True)

    project_dir = None
    results_button = QPushButton("Choose folder...", widget)

    @results_button.clicked.connect
    def choose_project_dir():
        nonlocal project_dir
        project_dir = str(
            QFileDialog.getExistingDirectory(
                results_button,
                "Output folder",
                expanduser("~"),
                QFileDialog.ShowDirsOnly,
            )
        )
        if not project_dir:
            return
        results_button.setText(basename(project_dir))
    
    button = QPushButton("Run", widget)
    @button.clicked.connect
    def run():
        def run_implementation(manager):
            engine = VinaDockingEngine(project_dir, manager)
            if continuation_check.isChecked():
                engine.run_docking(continuation=True)
            else:
                if not project_dir:
                    return
                #
                # Handle receptor
                #
                if tab_rec_idx == 0:
                    if not (validate_receptor_sel() and validate_box_sel()):
                        return
                    receptor_lib = saved_receptor_line.text().strip()
                    engine.prepare_receptor(
                        receptor_sel.currentText(),
                        box_sel.currentText(),
                        box_margin=box_margin_spin.value(),
                        allow_bad_res=allow_bad_res_check.isChecked(),
                        save_lib=receptor_lib
                    )
                elif tab_rec_idx == 1:
                    receptor_lib = rec_library_combo.currentText()
                    engine.prepare_receptor(from_lib=receptor_lib)
                #
                # Handle ligands
                #
                if tab_lig_idx == 0:
                    if not ligands_file:
                        return
                    engine.prepare_ligands(
                        ligands_path=ligands_file,
                        ph=ph_spin.value(),
                        cpu=cpu_spin.value(),
                        skip_acidbase=not enumerate_acidbase.isChecked(),
                        skip_tautomers=not enumerate_tautomers.isChecked(),
                        save_lib=compound_library_line.text().strip()
                    )
                elif tab_lig_idx == 1:
                    ligands_lib = library_combo.currentText().strip()
                    engine.prepare_ligands(from_lib=ligands_lib)
                #
                # Run docking
                #
                engine.run_docking(
                    scoring=function.currentText(),
                    exhaustiveness=exhaustiveness_spin.value(),
                    num_modes=num_modes_spin.value(),
                    min_rmsd=min_rmsd_spin.value(),
                    energy_range=energy_range_spin.value(),
                    cpu=cpu_spin.value(),
                    seed=seed_spin.value()
                )
        
        dialog = VinaThreadDialog(run_implementation)
        dialog.exec()

    horizontal_line1 = QFrame()
    horizontal_line1.setFrameShape(QFrame.HLine)
    horizontal_line1.setFrameShadow(QFrame.Sunken)

    horizontal_line2 = QFrame()
    horizontal_line2.setFrameShape(QFrame.HLine)
    horizontal_line2.setFrameShadow(QFrame.Sunken)

    horizontal_line3 = QFrame()
    horizontal_line3.setFrameShape(QFrame.HLine)
    horizontal_line3.setFrameShadow(QFrame.Sunken)

    #
    # setup layout
    #
    layout.setWidget(1, QFormLayout.SpanningRole, tab_receptor)
    layout.setWidget(2, QFormLayout.SpanningRole, horizontal_line1)

    layout.setWidget(3, QFormLayout.SpanningRole, tab_ligand)
    layout.setWidget(4, QFormLayout.SpanningRole, horizontal_line2)
    
    layout.setWidget(5, QFormLayout.SpanningRole, options_group)
    layout.setWidget(6, QFormLayout.SpanningRole, horizontal_line3)

    layout.addRow("Continuation:", continuation_check)
    layout.addRow("Output folder:", results_button)
    layout.addWidget(button)
    widget.setLayout(layout)

    return dockWidget


def __init_plugin__(app=None):
    
    run_widget = new_run_docking_widget()
    load_widget = new_load_results_widget()

    run_widget.hide()
    load_widget.hide()

    window = pymol.gui.get_qtwindow()
    window.addDockWidget(LeftDockWidgetArea, run_widget)
    window.addDockWidget(LeftDockWidgetArea, load_widget)

    def show_run_widget():
        run_widget.show()
    
    def show_load_widget():
        load_widget.show()

    from pymol.plugins import addmenuitemqt
    addmenuitemqt("XDrugPy::Docking Run", show_run_widget)
    addmenuitemqt("XDrugPy::Docking Analyze", show_load_widget)
