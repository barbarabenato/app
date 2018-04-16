import math
import pdb

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import numpy as np
from skimage import io, transform

from . import numpy_qt as npqt


class ToolPanel(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedWidth(180)
        self.setStyleSheet("background-color: #FF0000")


class ImagePanel(QGraphicsView):
    def __init__(self, parent):
        super().__init__(ImageScene(), parent)
        self.abs_scale_factor = 1.0
        # last x coord of the mouse pointer when clicking at mouse middle
        # button (wheel button) - used to move the scroll when pressing the
        # wheel and moving the mouse
        self.last_middle_click_x = 0
        self.last_middle_click_y = 0

        self.setMouseTracking(True) # enables the mouse tracking event



    def load_main_image(self, main_image_path):
        self.scene().clear()
        main_window = self.window()
        main_window.data.main_image = io.imread(main_image_path)

        qimg = npqt.numpy_to_qimage(main_window.data.main_image)
        self.scene().update_main_image(qimg)
        main_window.statusBar().showMessage('Loaded: ' + main_image_path)

    def regular_scale(self, scale_factor):
        self.abs_scale_factor *= scale_factor
        self.scale(scale_factor, scale_factor)

    @pyqtSlot()
    def normal_size(self):
        scale_factor = 1.0 / self.abs_scale_factor
        self.regular_scale(scale_factor)
        self.abs_scale_factor = 1.0


    @pyqtSlot()
    def fit_to_window(self):
        '''
        Fit the Main Image (if it exists) and fit it to the window (ImagePanel)
        '''

        '''
        Get the window (which is the top level QWidget in the hierarchy).
        Even passing the such window as the parent in the ImagePanel
        construction, it does not hold the MainWindow in our case,
        because the ImagePanel is compose in the centralWidget of the 
        MainWindow, then, the centralWidget becomes automatically the new 
        parent of the ImagePanel.
        '''
        main_window = self.window()
        main_image = main_window.data.main_image

        if main_image is not None:
            qimg = npqt.numpy_to_qimage(main_image)

            width_ratio = self.width() / qimg.width()
            height_ratio = self.height() / qimg.height()

            if width_ratio <= height_ratio:
                scale_factor = (self.width()-2) / qimg.width()
            else:
                scale_factor = (self.height()-2) / qimg.height()

            self.normal_size()
            self.regular_scale(scale_factor)


    def mousePressEvent(self, e: QMouseEvent):
        '''
        When pressing the middle button (wheel button), get the
        mouse pointer coordinate.
        When moving the mouse with such button pressed, it will compute the
        delta displacement to move the scroll bar .
        '''
        if e.buttons() == Qt.MidButton:
            self.last_middle_click_x = e.x()
            self.last_middle_click_y = e.y()
            self.setCursor(Qt.SizeAllCursor) # change the mouse pointer's icon

    def mouseReleaseEvent(self, e: QMouseEvent):
        '''
        If the middle button (wheel button) was released, set the mouse
        pointer's icon/shape to the original one.
        '''
        if e.button() == Qt.MidButton:
            self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, e: QMouseEvent):
        '''
        If self.setMouseTracking(False), this function only will be called
        when a button is pressed and moved. Otherwise, it will called after
        any mouse moviment.

        Since e.button() always returns value Qt.NoButton for mouse move events,
        we have to use e.buttons() instead.
        '''
        if e.buttons() == Qt.MidButton:
            # Move the scroll bar according to the mouse movement
            delta_x = (e.x() - self.last_middle_click_x)
            delta_y = (e.y() - self.last_middle_click_y)
            self.horizontalScrollBar().setValue(
                        self.horizontalScrollBar().value() - delta_x)
            self.verticalScrollBar().setValue(
                        self.verticalScrollBar().value() - delta_y)
            self.last_middle_click_x = e.x()
            self.last_middle_click_y = e.y()
        # print(e.pos())
        # scene_pos = self.mapToScene(e.pos())
        # print(scene_pos)
        # print(int(scene_pos.x()), int(scene_pos.y()))
        # print()


    def wheelEvent(self, e: QWheelEvent):
        '''
        When rolling up/down the mouse wheel on graphic view, this function
        is called.
        Here, we implement zoom in/out with the modifier ctrl.

        When overriding this e function, we lost its default behavior
        that was move the scroll bar.
        Then, we had to reimplement such actions.

        We could call QGraphicsView.wheelEvent(self, e) at the end to call
        the default behavior of scroll bar and other ones, but thus we cannot
        redefined them anymore, only what is not implemented.
        '''
        main_window = self.window()
        main_image = main_window.data.main_image

        if main_image is not None:
            # angle in that the mouse wheel is rotated in horizontal
            n_degrees_x = e.angleDelta().x()
            n_degrees_y = e.angleDelta().y() # vertical

            if n_degrees_x:
                self.horizontalScrollBar().setValue(
                    self.horizontalScrollBar().value() - n_degrees_x)
            if n_degrees_y:
                if e.modifiers() == Qt.ControlModifier:
                    self.zoom_in_out_by_wheel(e)
                elif e.modifiers() == Qt.ShiftModifier:
                    self.horizontalScrollBar().setValue(
                        self.horizontalScrollBar().value() - n_degrees_y)
                else:
                    self.verticalScrollBar().setValue(
                        self.verticalScrollBar().value() - n_degrees_y)


    def zoom_in_out_by_wheel(self, event: QWheelEvent):
        '''
        Zoom in/out the graphic scene by the mouse wheel moving.
        The zoom pursues the mouse pointer.
        '''

        # we have to remove the anchor behavior, in order "to be free" to
        # manipulate the scene as we want.
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)

        n_degrees = event.angleDelta().y()

        apply_scale = False
        scale_factor = 1.0
        main_window = self.window()

        if (n_degrees > 0) and (self.abs_scale_factor < 256.0):
            apply_scale = True
            scale_factor = 1.2
        elif (n_degrees < 0) and (self.abs_scale_factor > 0.05):
            apply_scale = True
            scale_factor = 0.8

        if apply_scale:
            '''
            The graphic view (GV) has its own coordinate space, starting at 0,
            0 on the left most corner.
            It has a scene centralized which also has its coordinate space.
            
            event.pos() gives the coords of the mouse pointer on GV's space.
            The function mapToScene then gives the coords of the mouse 
            pointer on the scene's space.
            '''
            # Returns the viewport coordinate point mapped to scene coordinates.
            # Save the scene pos
            old_pos = self.mapToScene(event.pos())

            self.regular_scale(scale_factor)
            main_window.menuBar().update_actions()

            new_pos = self.mapToScene(event.pos())
            # Move scene to old position
            delta = new_pos - old_pos
            self.translate(delta.x(), delta.y())


class ImageScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.canvas_image = None


    def update_main_image(self, qimg: QImage):
        self.clear()
        self.canvas_image = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
        self.addItem(self.canvas_image)


