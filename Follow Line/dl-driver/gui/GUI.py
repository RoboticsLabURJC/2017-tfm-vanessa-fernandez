#from gui.widgets.teleopWidget import TeleopWidget

__author__ = 'frivas'


from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets


class MainWindow(QtWidgets.QWidget):

    updGUI = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        ''' GUI class creates the GUI that we're going to use to
        preview the live video as well as the results of the real-time
        driving.
        '''

        QtWidgets.QWidget.__init__(self, parent)
        self.resize(900, 700)
        self.move(150, 50)
        self.setWindowIcon(QtGui.QIcon('gui/resources/jderobot.png'))

        self.updGUI.connect(self.updateGUI)

        # Original image label.
        self.camera1 = QtWidgets.QLabel(self)
        self.camera1.resize(450, 350)
        self.camera1.move(25, 50)
        self.camera1.show()

        # Prediction speeds label
        self.predict_v_label = QtWidgets.QLabel(self)
        self.predict_v_label.move(700, 100)
        self.predict_v_label.resize(100, 90)
        self.predict_v_label.show()

        self.predict_w_label = QtWidgets.QLabel(self)
        self.predict_w_label.move(700, 150)
        self.predict_w_label.resize(100, 90)
        self.predict_w_label.show()

        # Play button
        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.move(50, 500)
        self.pushButton.resize(300,50)
        self.pushButton.setText('Play Code')
        self.pushButton.clicked.connect(self.playClicked)
        self.pushButton.setCheckable(True)

        # Stop button
        icon = QtGui.QIcon('gui/resources/stop.png')
        self.stopButton = QtWidgets.QPushButton(self)
        self.stopButton.move(50, 600)
        self.pushButton.resize(300, 50)
        self.stopButton.setIcon(icon)
        self.stopButton.setText('Stop')
        self.stopButton.clicked.connect(self.stopClicked)

        # Logo
        self.logo_label = QtWidgets.QLabel(self)
        self.logo_label.resize(100, 100)
        self.logo_label.move(700, 500)
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
        self.predict_v_label.setText("%d v" % (50))
        self.predict_w_label.setText("%d w" % (2))

    def getCamera(self):
        return self.camera

    def setCamera(self,camera):
        self.camera=camera

    def getMotors(self):
        return self.motors

    def setMotors(self,motors):
        self.motors=motors

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
        #print ("newX: %f, newY: %f" % (newX, newY) )
        myW=-newX*self.motors.getMaxW()
        myV=-newY*self.motors.getMaxV()
        self.motors.sendV(myV)
        self.motors.sendW(myW)
        None

    def stopClicked(self):
        self.motors.sendV(0)
        self.motors.sendW(0)
        #self.teleop.returnToOrigin()

    def closeEvent(self, event):
        self.algorithm.kill()
        self.camera.stop()
        event.accept()
