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


class ToolPanel(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedWidth(180)
        self.setStyleSheet("background-color: #FF0000")


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        #self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass
    

class matplotlibWidget(QWidget):

    def __init__( self, parent=None, width=5, height=4, dpi=100):
        QWidget.__init__( self, parent )
        self.canvas = MyMplCanvas() #create canvas that will hold our plot
        self.navi_toolbar = NavigationToolbar(self.canvas, self) #createa navigation toolbar for our plot canvas

        self.vbl = QVBoxLayout()
        self.vbl.addWidget( self.canvas )
        self.vbl.addWidget(self.navi_toolbar)
        self.setLayout( self.vbl )


class MyDynamicPlot(matplotlibWidget):
    """A canvas that updates itself every second with a new plot."""
    def __init__(self, *args, **kwargs):
        matplotlibWidget.__init__(self, *args, **kwargs)
        self.abs_scale_factor = 1.0
        #timer = QtCore.QTimer(self)
        #timer.timeout.connect(self.update_figure)
        #timer.start(1000)
        self.data = 0
        self.feats_2d = 0

    def get_data(self, data_path):
        self.data = ift._read_dataset(data_path)

        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
        self.feats_2d = tsne.fit_transform(self.data['feats'])

    def cmap(self):
        my_cmap = matplotlib.cm.get_cmap('Set3')
        my_cmap.set_under('k')
        return my_cmap

    def compute_initial_figure(self, data_path):
        main_window = self.window()
        #main_window.data.main_image = io.imread(main_image_path)

        self.get_data(data_path)

        self.canvas.axes.scatter(self.feats_2d[:,0], self.feats_2d[:,1], c=np.ma.ravel(self.data['truelabel']), s=20, cmap=self.cmap())
        self.canvas.draw()

        main_window.statusBar().showMessage('Loaded: ' + data_path)

        #self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        l = [random.randint(0, 10) for i in range(4)]

        #self.axes.plot([0, 1, 2, 3], l, 'r')
        self.canvas.axes.scatter(self.feats_2d[:,0], self.feats_2d[:,1], c=np.ma.ravel(self.data['truelabel']), s=20, cmap=self.cmap())
        self.canvas.draw()


class ImageScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.canvas_image = None


    def update_main_image(self, qimg: QImage):
        self.clear()
        self.canvas_image = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
        self.addItem(self.canvas_image)


