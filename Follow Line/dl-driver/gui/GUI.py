from gui.widgets.teleopWidget import TeleopWidget

__author__ = 'frivas'


from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow
#from gui.form import Ui_MainWindow
from gui.widgets.cameraWidget import CameraWidget
from gui.widgets.logoWidget import LogoWidget



class MainWindow(QtWidgets.QWidget):

    updGUI = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        ''' GUI class creates the GUI that we're going to use to
        preview the live video as well as the results of the real-time
        driving.
        '''
        QtWidgets.QWidget.__init__(self, parent)
        #self.setupUi(self)
        self.teleop=TeleopWidget(self)
        self.tlLayout.addWidget(self.teleop)
        self.teleop.setVisible(True)
        self.logo = LogoWidget(self)
        self.logoLayout.addWidget(self.logo)
        self.logo.setVisible(True)

        self.pushButton.clicked.connect(self.playClicked)
        self.pushButton.setCheckable(True)
        self.updGUI.connect(self.updateGUI)
        self.camera1=CameraWidget(self)

        self.stopButton.clicked.connect(self.stopClicked)

    def updateGUI(self):
        #print 'update gui'
        self.camera1.updateImage()

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

    def setAlgorithm(self, algorithm ):
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
        self.teleop.returnToOrigin()

    def closeEvent(self, event):
        self.algorithm.kill()
        self.camera.stop()
        event.accept()
