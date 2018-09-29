
# Based on @frivas
__author__ = 'vmartinezf'


from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets


class MainWindow(QtWidgets.QWidget):

    updGUI = QtCore.pyqtSignal()
    stopSIG = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        ''' GUI class creates the GUI that we're going to use to
        preview the live video as well as the results of the real-time
        driving.
        '''

        QtWidgets.QWidget.__init__(self, parent)
        self.resize(800, 1000)
        self.move(150, 50)
        self.setWindowIcon(QtGui.QIcon('gui/resources/jderobot.png'))

        self.updGUI.connect(self.updateGUI)

        # Original image label.
        self.camera1 = QtWidgets.QLabel(self)
        self.camera1.resize(450, 350)
        self.camera1.move(50, 50)
        self.camera1.show()

        # Prediction speeds label
        self.predict_v_label = QtWidgets.QLabel(self)
        self.predict_v_label.move(600, 100)
        self.predict_v_label.resize(100, 90)
        self.predict_v_label.show()

        self.predict_w_label = QtWidgets.QLabel(self)
        self.predict_w_label.move(600, 150)
        self.predict_w_label.resize(100, 90)
        self.predict_w_label.show()

        # Play button
        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.move(50, 450)
        self.pushButton.resize(450,50)
        self.pushButton.setText('Play Code')
        self.pushButton.clicked.connect(self.playClicked)
        self.pushButton.setCheckable(True)

        # Teleoperator
        self.line = QtCore.QPointF(0, 0)
        self.qimage = QtGui.QImage()
        self.qimage.load('gui/resources/ball.png')
        self.stopSIG.connect(self.stop)
        self.layout = QtWidgets.QLabel()
        self.widthTeleop = 350
        self.heightTeleop = 350
        self.pointxTeleop = 100
        self.pointyTeleop = 550
        self.layout.move(self.pointxTeleop, self.pointyTeleop)
        self.layout.resize(self.widthTeleop, self.heightTeleop)

        # Stop button
        icon = QtGui.QIcon('gui/resources/stop.png')
        self.stopButton = QtWidgets.QPushButton(self)
        self.stopButton.move(100, 925)
        self.stopButton.resize(350, 50)
        self.stopButton.setIcon(icon)
        self.stopButton.setText('Stop')
        self.stopButton.clicked.connect(self.stopClicked)

        # Logo
        self.logo_label = QtWidgets.QLabel(self)
        self.logo_label.resize(100, 100)
        self.logo_label.move(600, 700)
        self.logo_label.setScaledContents(True)

        logo_img = QtGui.QImage()
        logo_img.load('gui/resources/jderobot.png')
        self.logo_label.setPixmap(QtGui.QPixmap.fromImage(logo_img))
        self.logo_label.show()

    def updateGUI(self):
        ''' Updates the GUI for every time the thread change '''
        # We get the original image and display it.

        self.im_prev = self.camera.getImage()
        im = QtGui.QImage(self.im_prev.data, self.im_prev.data.shape[1], self.im_prev.data.shape[0],
                          QtGui.QImage.Format_RGB888)
        self.im_scaled = im.scaled(self.camera1.size())

        self.camera1.setPixmap(QtGui.QPixmap.fromImage(self.im_scaled))

        # We get the v and w
        newfont = QtGui.QFont("Times", 18, QtGui.QFont.Bold)
        self.predict_v_label.setFont(newfont)
        self.predict_w_label.setFont(newfont)
        self.predict_v_label.setText("%d v" % (0))
        self.predict_w_label.setText("%d w" % (0))

    def getCamera(self):
        return self.camera

    def setCamera(self,camera):
        self.camera=camera

    def getMotors(self):
        return self.motors

    def setMotors(self,motors):
        self.motors=motors

    def stop(self):
        self.line = QtCore.QPointF(0, 0)
        self.repaint()

    def mouseMoveEvent(self, e):
        if e.buttons() == QtCore.Qt.LeftButton:
            x = e.x() - self.widthTeleop / 2
            y = e.y() - self.heightTeleop / 2
            self.line = QtCore.QPointF(x, y)
            self.repaint()

    def returnToOrigin(self):
        x = 0
        y = 0
        self.line = QtCore.QPointF(x, y)
        self.repaint()

    def paintEvent(self, e):
        _width = self.widthTeleop
        _height = self.heightTeleop

        width = 2

        painter = QtGui.QPainter(self)

        # Background
        painter.fillRect(self.pointxTeleop, self.pointyTeleop, self.widthTeleop, self.heightTeleop, QtCore.Qt.black)

        # Lines

        pen = QtGui.QPen(QtCore.Qt.blue, width)
        painter.setPen(pen)
        painter.translate(QtCore.QPoint(self.pointxTeleop + _width/2, self.pointyTeleop + _height/2))

        # Axis
        painter.drawLine(QtCore.QPointF(-_width/2, 0),
                         QtCore.QPointF(_width/2, 0))

        painter.drawLine(QtCore.QPointF(0, -_height/2),
                         QtCore.QPointF(0, _height/2))

        # With mouse
        pen = QtGui.QPen(QtCore.Qt.red, width)
        painter.setPen(pen)

        # We check if mouse is in the limits
        if abs(self.line.x() * 2) >= self.widthTeleop:
            if self.line.x() >= 0:
                self.line.setX(self.widthTeleop / 2)
            elif self.line.x() < 0:
                self.line.setX((-self.widthTeleop / 2) + 1)

        if abs(self.line.y() * 2) >= self.heightTeleop:
            if self.line.y() >= 0:
                self.line.setY(self.heightTeleop / 2)
            elif self.line.y() < 0:
                self.line.setY((-self.heightTeleop / 2) + 1)

        painter.drawLine(QtCore.QPointF(self.line.x(), -_width/2),
                         QtCore.QPointF(self.line.x(), _width/2))

        painter.drawLine(QtCore.QPointF(-_height/2, self.line.y()),
                         QtCore.QPointF(_height/2, self.line.y()))

        v_normalized = (1.0 / (self.heightTeleop / 2)) * self.line.y()
        v_normalized = float("{0:.2f}".format(v_normalized))
        w_normalized = (1.0 / (self.widthTeleop / 2)) * self.line.x()
        w_normalized = float("{0:.2f}".format(w_normalized))

        self.setXYValues(w_normalized, v_normalized)
        painter.drawImage(self.line.x() - self.qimage.width() / 2, self.line.y() - self.qimage.height() / 2, self.qimage)

    def playClicked(self):
        if self.pushButton.isChecked():
            self.pushButton.setText('Stop Code')
            self.pushButton.setStyleSheet("background-color: #7dcea0")
            self.algorithm.play()
        else:
            self.pushButton.setText('Play Code')
            self.pushButton.setStyleSheet("background-color: #ec7063")
            self.algorithm.stop()

    def setAlgorithm(self, algorithm):
        self.algorithm=algorithm

    def getAlgorithm(self):
        return self.algorithm

    def setXYValues(self,newX,newY):
        myW=-newX*self.motors.getMaxW()
        myV=-newY*self.motors.getMaxV()
        self.motors.sendV(myV)
        self.motors.sendW(myW)
        None

    def stopClicked(self):
        self.motors.sendV(0)
        self.motors.sendW(0)
        self.returnToOrigin()

    def closeEvent(self, event):
        self.algorithm.kill()
        self.camera.stop()
        event.accept()
