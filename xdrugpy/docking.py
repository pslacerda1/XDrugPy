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

from .utils import LIGAND_LIBRARIES_DIR, TEMPDIR, run, dendrogram, RECEPTOR_LIBRARIES_DIR


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

class BaseThread(QThread):
    vinaStarted = pyqtSignal()

    numSteps = pyqtSignal(int)
    incrementStep = pyqtSignal()

    logEvent = pyqtSignal(str)
    logCodeEvent = pyqtSignal(str)
    logRawEvent = pyqtSignal(str)

    done = pyqtSignal(bool)

    def __init__(self, *args, parent=None):
        super().__init__(parent)
        self.args = args


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
    # pm.split_states('lig')
    # pm.set_name(f'lig_{mode.zfill(4)}', 'lig')
    pm.delete('lig_*')
    pm.alter('lig', 'chain="Z"')
    pm.alter('lig', 'resn="LIG"')
    pm.alter('lig', 'resi=1')

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
    plt.xticks(rotation=45)
    plt.show()

    fig, ax = plt.subplots(layout="constrained")
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
    
    D = []
    for idx1, mol1 in enumerate(mols):
        for idx2, mol2 in enumerate(mols):
            if idx1 >= idx2:
                continue
            d = euclidean(mol1, mol2)
            D.append(d)
    dendrogram(
        D,
        labels=labels,
        orientation='right',
        color_threshold=-1,
        distance_sort=True,
        ax=ax
    )
    ax.set_xlim(0)
    plt.show()


class OrderedCounter(Counter, OrderedDict):
    '''
    Counter that remembers the order elements are first encountered
    https://stackoverflow.com/a/23747652/199332
    '''


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
            QFileDialog.getOpenFileName(
                show_table_button,
                "Docking file",
                expanduser("~"),
                "Docking file (docking.json)",
            )[0]
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

class VinaThreadDialog(QDialog):

    def __init__(self, *vina_args, parent=None):
        super().__init__(parent)
        self.vina = VinaThread(*vina_args)
        self.vina.done.connect(self._done)
        
        project_dir = vina_args[0]
        self.progress_thread = ProgressThread(
            ligands_dir="%s/queue" % project_dir,
            output_dir="%s/output" % project_dir,
        )
        self.progress_thread.incrementStep.connect(self.vina.incrementStep.emit)
        self.progress_thread.numSteps.connect(self.vina.numSteps.emit)
        
        # self.progress_thread.finished.connect()
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
        @self.vina.incrementStep.connect
        def incrementStep():
            self.progress.setValue(self.progress.value() + 1)

        # Rich text output
        self.text = QTextEdit(self)
        self.layout.addWidget(self.text)
        self.text.setReadOnly(True)
        self.vina.logEvent.connect(self._appendHtml)
        self.vina.logCodeEvent.connect(self._appendCodeHtml)

        # Ok / Cancel buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Abort, QtCore.Qt.Horizontal, self
        )
        self.layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self._done)
        self.button_box.rejected.connect(self._abort)
        self.button_box.button(QDialogButtonBox.Ok).setDisabled(True)

        # Start docking
        self.vina.start()
        @self.vina.vinaStarted.connect
        def vinaStarted():
            self.progress_thread.start()

    def _appendHtml(self, html):
        self.text.moveCursor(QTextCursor.End)
        self.text.insertHtml(self._prepareHtml(html))

    def _appendCodeHtml(self, html):
        self.text.moveCursor(QTextCursor.End)
        self.text.insertHtml("<pre>" + self._prepareHtml(html) + "</pre>")

    def _abort(self):
        self.vina.terminate()
        self.progress_thread.terminate()
        self.done(QDialog.Rejected)

    def _done(self, success):
        ok_button = self.button_box.button(QDialogButtonBox.Ok)
        abort_button = self.button_box.button(QDialogButtonBox.Abort)

        ok_button.setDisabled(False)
        abort_button.setDisabled(True)

        @self.button_box.accepted.connect
        def _done():
            if success:
                self.accept()
            else:
                self.reject()

    @staticmethod
    def _prepareHtml(html):
        return textwrap.dedent(html)


#
# Run docking software
#
class ProgressThread(QThread):
    numSteps = pyqtSignal(int)
    incrementStep = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, ligands_dir, output_dir):
        self.queue_dir = ligands_dir
        self.output_dir = output_dir
        self.ligands_done = set()
        super().__init__()

    def run(self):
        while True:
            sleep(1)
            ligands = glob(f"{self.queue_dir}/*.pdbqt")
            for ligand_pdbqt in ligands:
                lig = basename(ligand_pdbqt)[:-6]
                output_pdbqt = f"{self.output_dir}/{lig}_out.pdbqt"
                if lig not in self.ligands_done and exists(output_pdbqt):
                    self.ligands_done.add(lig)
                    os.unlink(ligand_pdbqt)
                    self.incrementStep.emit()
            if len(ligands) == glob(f"{self.output_dir}/*_out.dbqt"):
                self.finished.emit()


class VinaThread(BaseThread):
    def run(self):
        (
            project_dir,
            ligands_file,
            saved_receptor,
            receptor_sel,
            box_sel,
            box_margin,
            allow_errors,
            ph,
            exhaustiveness,
            num_modes,
            min_rmsd,
            energy_range,
            cpu,
            seed,
            save_library_check,
            library,
            function,
        ) = self.args

        #
        # Check previous output
        #
        if os.listdir(project_dir):
            self.logEvent.emit(f"""
                <br/>
                <font color="red">
                    <b>The docking folder is not empty: '{project_dir}'</b>
                </font>
            """)

        #
        # Prepare receptor
        #
        receptor_lib = RECEPTOR_LIBRARIES_DIR + saved_receptor + '.pdbqt'
        receptor_pdbqt = f"{project_dir}/receptor.pdbqt"
        receptor_pdb = f"{project_dir}/receptor.pdb"
        if saved_receptor and exists(receptor_lib):
            shutil.copy(receptor_lib, receptor_pdbqt)
            self.logEvent.emit(f"""
                <br/>
                <br/><b>Recovered stored receptor:</b> {receptor_lib}
            """)
            
        else:
            pm.save(receptor_pdb, receptor_sel)
            command = (
                f'python -m meeko.cli.mk_prepare_receptor --read_pdb "{receptor_pdb}" -p "{receptor_pdbqt}"'
            )
            if allow_errors:
                command = f"{command} -a"
            self.logEvent.emit(f"""
                <br/>
                <br/><b>Preparing receptor.</b>
                <br/><b>Command:</b> {command}
                <br/>
            """)
            output, success = run(command)
            self.logCodeEvent.emit(output)
            if not success:
                self.done.emit(False)
                return
            
            if saved_receptor:
                shutil.copy(receptor_pdbqt, receptor_lib)   
        #
        # Compute box variables
        #
        box_lib = RECEPTOR_LIBRARIES_DIR + '/' + saved_receptor + '.box'
        if saved_receptor and exists(box_lib):
            with open(box_lib) as box_file:
                size = box_file.readline().split()
                size_x, size_y, size_z = map(float, size)
                
                center = box_file.readline().split()
                center_x, center_y, center_z = map(float, center)
            
        else:

            box_coords = pm.get_coords(box_sel)

            max = np.max(box_coords, axis=0)
            min = np.min(box_coords, axis=0)

            half_size = (max - min) / 2
            center = min + half_size

            size_x, size_y, size_z = (half_size + box_margin) * 2
            center_x, center_y, center_z = center

            size_x, size_y, size_z = (
                round(float(size_x), 2),
                round(float(size_y), 2),
                round(float(size_z), 2),
            )

            center_x, center_y, center_z = (
                round(float(center_x), 2),
                round(float(center_y), 2),
                round(float(center_z), 2),
            )
            with open(box_lib, 'w') as box_file:
                box_file.write(f"{size_x} {size_y} {size_z}\n")
                box_file.write(f"{center_x} {center_y} {center_z}")
        #
        # Prepare ligands
        #
        queue_dir = project_dir + "/queue"

        if library:
            library_dir = LIGAND_LIBRARIES_DIR + '/' + library
            try:
                if exists(queue_dir):
                    shutil.rmtree(queue_dir)
            except OSError:
                os.unlink(queue_dir)
            shutil.copytree(library_dir, queue_dir)
            self.logEvent.emit(f"""
                <br/>
                <br/><b>Recovered stored library:</b> {library_dir}
            """)
        elif ligands_file:
            if exists(queue_dir):
                try:
                    shutil.rmtree(queue_dir)
                except OSError:
                    os.unlink(queue_dir)
            
            #
            # Scrubbe isomers
            #
            ligands_sdf = project_dir + "/ligands.sdf"
            command = (
                f'python -m scrubber.main -o "{ligands_sdf}" --ph {ph} --cpu {cpu} "{ligands_file}"'
            )
            self.logEvent.emit(
                f"""
                    <br/>
                    <br/><b>Scrubbing ligands.</b>
                    <br/><b>Command:</b> {command}
                    <br/>
                """
            )
            output, success = run(command)
            self.logCodeEvent.emit(output)
            if not success:
                self.done.emit(False)
                return

            #
            # Converting to PDBQT
            #
            if not exists(queue_dir):
                os.makedirs(queue_dir)
            command = (
                f'python -m meeko.cli.mk_prepare_ligand -i "{ligands_sdf}" --multimol_outdir "{queue_dir}"'
            )
            self.logEvent.emit(
                f"""
                    <br/>
                    <br/><b>Converting ligands to PDBQT.</b>
                    <br/><b>Command:</b> {command}
                    <br/>
                """
            )
            output, success = run(command)
            self.logCodeEvent.emit(output)

            if save_library_check:
                library_dir = splitext(basename(ligands_file))[0]
                library_dir = LIGAND_LIBRARIES_DIR + '/' + library_dir
                self.logEvent.emit(f"""
                    <br/>
                    <br/><b>Storing compound library at:</b> {library_dir}
                """)
                try:
                    shutil.rmtree(library_dir)
                except:
                    pass
                shutil.copytree(queue_dir, library_dir)

        self.numSteps.emit(len(os.listdir(queue_dir)))
        #
        # Create Vina results directory
        #
        output_dir = f"{project_dir}/output"
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass

        try:
            os.mkdir(queue_dir)
        except FileExistsError:
            pass
        
        
        project_file = project_dir + "/docking.json"
        project_data = {
            "function": function,
            "box_sel": box_sel,
            "box_margin": box_margin,
            "size_x": size_x,
            "size_y": size_y,
            "size_z": size_z,
            "center_x": center_x,
            "center_y": center_y,
            "center_z": center_z,
            "ph": ph,
            "exhaustiveness": exhaustiveness,
            "seed": seed
        }

        command = (
            f"vina"
            f" --receptor '{receptor_pdbqt}'"
            f" --scoring {function}"
            f" --cpu {cpu}"
            f" --seed {seed}"
            f" --size_x {size_x}"
            f" --size_y {size_y}"
            f" --size_z {size_z}"
            f" --center_x {center_x}"
            f" --center_y {center_y}"
            f" --center_z {center_z}"
            f" --exhaustiveness {exhaustiveness}"
            f" --num_modes {num_modes}"
            f" --min_rmsd {min_rmsd}"
            f" --energy_range {energy_range}"
            f" --dir '{output_dir}'"
            f" --batch '{queue_dir}/'*.pdbqt"
        )
        self.logEvent.emit(f"""    
            <br/>
            <br/><b>Docking ligands.</b>
            <br/><b>Command:</b> {command}
        """)
        project_file = f"{project_dir}/docking.json"
        with open(project_file, "w") as docking_file:
            json.dump(project_data, docking_file, indent=4)
        
        self.vinaStarted.emit()
        output, success = run(command)

        @self.finished.connect
        def finished():
            output_ligands = len(glob(f"{output_dir}/*_out.pdbqt"))
            n_ligands = len(glob(f"{queue_dir}/*.pdbqt"))
            self.logEvent.emit("<br/><h2>Summary</h2>")
            summary = f"""
                <br/><b>Total expected:</b> {n_ligands + output_ligands}
                <br/><b>Total done:</b> {output_ligands}
            """
            if output_ligands < n_ligands:
                self.logEvent.emit(f"<font color='red'>{summary}</font>")
            else:
                self.logEvent.emit(f"{summary}")

        self.done.emit(True)


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

    @box_margin_spin.valueChanged.connect
    def display_box(margin):
        pm.delete("box")
        display_box_sel("box", box_sel.currentText(), margin)


    allow_errors_check = QCheckBox(widget)
    allow_errors_check.setChecked(False)
    tab1_layout.addRow("Allow Meeko errors:", allow_errors_check)

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
    tab1_layout.addRow("Ligand file:", ligands_button)

    ph_spin = QDoubleSpinBox(widget)
    ph_spin.setRange(0.0, 14.0)
    ph_spin.setValue(7.0)
    ph_spin.setSingleStep(0.1)
    ph_spin.setDecimals(1)
    tab1_layout.addRow("Ligands pH:", ph_spin)

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
    save_library_check = QCheckBox()
    save_library_check.setChecked(False)
    tab1_layout.addRow("Store library:", save_library_check)


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

    #
    # Scoring function
    #
    function = QComboBox(widget)
    function.addItems(["vina", "vinardo"])

    exhaustiveness_spin = QSpinBox(widget)
    exhaustiveness_spin.setRange(1, 50)
    exhaustiveness_spin.setValue(8)

    num_modes_spin = QSpinBox(widget)
    num_modes_spin.setRange(1, 100)
    num_modes_spin.setValue(3)

    min_rmsd_spin = QDoubleSpinBox(widget)
    min_rmsd_spin.setRange(0.0, 3.0)
    min_rmsd_spin.setValue(1.0)

    energy_range_spin = QDoubleSpinBox(widget)
    energy_range_spin.setRange(0, 10.0)
    energy_range_spin.setValue(3.0)

    cpu_count = QThread.idealThreadCount()
    cpu_spin = QSpinBox(widget)
    cpu_spin.setRange(1, cpu_count)
    cpu_spin.setValue(cpu_count)

    seed_spin = QSpinBox(widget)
    seed_spin.setRange(1, 10000)
    seed_spin.setValue(1)

    #
    # Choose  output folder
    #
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
        nonlocal ligands_file

        if not project_dir:
            return
        
        if tab_rec_idx == 0:
            if not (validate_receptor_sel() & validate_box_sel()):
                return
            receptor_lib = saved_receptor_line.text().strip()
                
        elif tab_rec_idx == 1:
            receptor_lib = rec_library_combo.currentText()
        
        library = library_combo.currentText().strip()
        if tab_lig_idx == 0:
            if not ligands_file:
                return
            library = None
        elif tab_lig_idx == 1:
            if not library:
                return
            ligands_file = None

        dialog = VinaThreadDialog(
            project_dir,
            ligands_file,
            receptor_lib,
            receptor_sel.currentText(),
            box_sel.currentText(),
            box_margin_spin.value(),
            allow_errors_check.isChecked(),
            ph_spin.value(),
            exhaustiveness_spin.value(),
            num_modes_spin.value(),
            min_rmsd_spin.value(),
            energy_range_spin.value(),
            cpu_spin.value(),
            seed_spin.value(),
            save_library_check.isChecked(),
            library,
            function.currentText(),
        )
        dialog.exec_()

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

    layout.addRow("Function:", function)
    layout.addRow("Exhaustiveness:", exhaustiveness_spin)
    layout.addRow("Number of modes:", num_modes_spin)
    layout.addRow("Minimum RMSD:", min_rmsd_spin)
    layout.addRow("Energy range:", energy_range_spin)
    layout.addRow("Number of CPUs:", cpu_spin)
    layout.addRow("Seed number:", seed_spin)
    
    layout.setWidget(12, QFormLayout.SpanningRole, horizontal_line3)
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
