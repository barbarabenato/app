import pdb

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from .menu_bar import *
from .main_panel import ToolPanel, MyDynamicPlot



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
        self.image_panel = None

        self.init_ui()

        self.show()


    def init_ui(self):
        self.setCentralWidget(QWidget())
        self.set_main_window_general_properties()

        self.tool_panel = ToolPanel(self)
        self.image_panel = MyDynamicPlot(self, width=5, height=4, dpi=100)

        # creates a layout and set it to the centralWidget
        main_layout = QGridLayout(self.centralWidget())
        main_layout.addWidget(self.tool_panel, 0, 0)
        main_layout.addWidget(self.image_panel, 0, 1)

        # since our MenuBar connects some functions/slots from other files
        # and objects to actions, we load it at the end
        self.setMenuBar(MenuBar(self)) # self (MainWindow) is the parent


    def set_main_window_general_properties(self):
        self.setWindowTitle("Label Propagation Interface")
        # self.setMinimumSize(800, 500)
        self.setGeometry(150, 45, 900, 700)
        self.statusBar().showMessage('Ready')



