import pdb

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from .menu_bar import *
from .main_panel import ToolPanel, tSNEPanel, neighboursPanel



class Data():
    def __init__(self, main_image=None, grad_image=None, label_image=None):
        self.main_image = main_image
        self.grad_image = grad_image
        self.label_image = label_image


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = Data()

        self.tool_panel = None
        self.tsne_panel = None
        self.image_panel = None

        self.init_ui()

        self.show()


    def init_ui(self):
        self.setCentralWidget(QWidget())
        self.set_main_window_general_properties()

        self.tool_panel = ToolPanel(self)
        self.tsne_panel = tSNEPanel(self)
        self.neigh_panel = neighboursPanel(self)

        # creates a layout and set it to the centralWidget
        main_layout = QGridLayout(self.centralWidget())
        main_layout.addWidget(self.tool_panel, 0, 0)
        main_layout.addWidget(self.tsne_panel, 0, 1)
        main_layout.addWidget(self.neigh_panel, 0, 2)

        # since our MenuBar connects some functions/slots from other files
        # and objects to actions, we load it at the end
        self.setMenuBar(MenuBar(self)) # self (MainWindow) is the parent


    def set_main_window_general_properties(self):
        self.setWindowTitle("Label Propagation Interface")
        # self.setMinimumSize(800, 500)
        self.setGeometry(200, 200, 1200, 600)
        self.statusBar().showMessage('Ready')



