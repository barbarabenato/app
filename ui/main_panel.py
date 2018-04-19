import math
import pdb

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


import random
import matplotlib
matplotlib.use("Qt5Agg")
from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from sklearn.manifold import TSNE


import numpy as np
from skimage import io, transform

import ift_dataset as ift


class ToolPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedWidth(150)

        self.title = None
        self.color_button = None
        self.select_button = None
        self.propag_button = None
        self.classif_button = None

        self.init_ui()

        self.show()


    def init_ui(self):

        self.title = QLabel('Tools')

        self.color_button = QPushButton("Color selection")
        self.select_button = QPushButton("Selection Tool")
        self.propag_button = QPushButton("Do propagation")
        self.classif_button = QPushButton("Classify")

        tool_layout = QGridLayout()
        tool_layout.addWidget(self.color_button, 2, 0)
        tool_layout.addWidget(self.select_button, 3, 0)
        tool_layout.addWidget(self.propag_button, 4, 0)
        tool_layout.addWidget(self.classif_button, 5, 0)

        vbox = QGridLayout()
        vbox.addLayout(tool_layout, 1,0)

        self.setLayout(vbox)
        

class tSNEPanel(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedWidth(500)
        self.setStyleSheet("background-color: #FFFFFF")
        
        self.data = None


class neighboursPanel(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedWidth(500)
        self.setStyleSheet("background-color: #FFFFFF")
        
        self.data = None


class ImageScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.canvas_image = None


    def update_main_image(self, qimg: QImage):
        self.clear()
        self.canvas_image = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
        self.addItem(self.canvas_image)


