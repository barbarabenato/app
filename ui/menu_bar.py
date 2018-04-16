import os
import pdb

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class FileDialogControl():
    first_dialog_main_image = True


class FileMenu(QMenu):
    '''
    A window is a widget that isn't visually the child of any other widget and
    that usually has a frame and a window title.
    A window can have a parent widget. It will then be grouped with its parent
    and deleted when the parent is deleted, minimized when the parent is
    minimized etc.
    If supported by the window manager, it will also have a common taskbar
    entry with its parent.

    QMenu is a window by default, even if a parent widget is specified in the
    constructor.

    In our case, the FileMenu (which is QMenu) has the a QMenuBar as parent.
    self.parent() gives us the QMenuBar object.
    We would expect that self.window() returns the MainWindow object, which is
    parent of the MenuBar, but it returns the own self, since QMenu is a window.
    '''

    def __init__(self, menu_bar):
        super().__init__(menu_bar)
        self.setTitle("&File")

        self.open_act = QAction("&Open Data", menu_bar)
        # QKeySequence.Open gets the default open shortcut of the current OS
        # self.open_act.setShortcut(QKeySequence.Open)
        self.open_act.triggered.connect(self.openFileNameDialog)
        self.addAction(self.open_act)

        self.addSeparator()

        self.save_act = QAction("&Save Data", menu_bar)
        # QKeySequence.Open gets the default open shortcut of the current OS
        # self.open_act.setShortcut(QKeySequence.Open)
        self.save_act.triggered.connect(self.saveFileDialog)
        self.addAction(self.save_act)

        self.addSeparator()

        '''
        When using PyQt on a Mac Os, the system will intercept certain
        commands contain the word 'Quit', 'Exit', 'Setting', 'Settings',
        'preferences' and probably a slew of others, and remove them from your
        menubar because they are reserved labels.
        If a menubar header has no items, it will not display, making it
        appear as if you haven't modified the menubar.
        '''
        exit_act = self.addAction("&Close App",
                                       QApplication.instance().quit,
                                       QKeySequence.Close)

    @pyqtSlot()

    def openFileNameDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Opening a OPF dataset...", "","Zip Files (*.zip);;All Files (*)", options=options)
        if fileName:
            print(fileName)

            menu_bar = self.parent()
            main_window = menu_bar.window()
            main_window.image_panel.compute_initial_figure(fileName)#main_image_path)


    def saveFileDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Saving a OPF dataset...","","Zip Files (*.zip);;All Files (*)", options=options)
        if fileName:
            print(fileName)



class MenuBar(QMenuBar):
    def __init__(self, main_window):
        # the parent is the main_window
        super().__init__(main_window)

        self.file_menu = FileMenu(self)

        self.addMenu(self.file_menu)

    def update_actions(self):
        self.view_menu.update_actions()