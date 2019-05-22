# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'recordUI.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import QApplication, QDialog

class Ui_Record(object):
    def setupUi(self, Record):
        Record.setObjectName("Record")
        Record.resize(358, 490)
        self.groupBox = QtWidgets.QGroupBox(Record)
        self.groupBox.setGeometry(QtCore.QRect(0, 0, 371, 461))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.FrameLabel = QtWidgets.QLabel(self.groupBox)
        self.FrameLabel.setGeometry(QtCore.QRect(0, -10, 361, 361))
        self.FrameLabel.setStyleSheet("background:black;")
        self.FrameLabel.setObjectName("FrameLabel")
        self.DialogBox = QtWidgets.QDialogButtonBox(self.groupBox)
        self.DialogBox.setGeometry(QtCore.QRect(80, 420, 156, 23))
        self.DialogBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.DialogBox.setObjectName("DialogBox")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(0, 0, 371, 461))
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.FrameLabel_2 = QtWidgets.QLabel(self.groupBox_2)
        self.FrameLabel_2.setGeometry(QtCore.QRect(0, -10, 361, 361))
        self.FrameLabel_2.setStyleSheet("background:black;")
        self.FrameLabel_2.setObjectName("FrameLabel_2")
        self.DialogBox_2 = QtWidgets.QDialogButtonBox(self.groupBox_2)
        self.DialogBox_2.setGeometry(QtCore.QRect(80, 420, 156, 23))
        self.DialogBox_2.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.DialogBox_2.setObjectName("DialogBox_2")
        self.layoutWidget = QtWidgets.QWidget(self.groupBox_2)
        self.layoutWidget.setGeometry(QtCore.QRect(80, 370, 165, 48))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 0, 0, 1, 1)
        self.NameLineEdit_2 = QtWidgets.QLineEdit(self.layoutWidget)
        self.NameLineEdit_2.setObjectName("NameLineEdit_2")
        self.gridLayout_2.addWidget(self.NameLineEdit_2, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 1, 0, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout_2.addWidget(self.lineEdit_2, 1, 1, 1, 1)
        self.widget = QtWidgets.QWidget(self.groupBox)
        self.widget.setGeometry(QtCore.QRect(80, 370, 165, 48))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.NameLineEdit = QtWidgets.QLineEdit(self.widget)
        self.NameLineEdit.setObjectName("NameLineEdit")
        self.gridLayout.addWidget(self.NameLineEdit, 0, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 1, 1, 1, 1)

        self.retranslateUi(Record)
        QtCore.QMetaObject.connectSlotsByName(Record)

    def retranslateUi(self, Record):
        _translate = QtCore.QCoreApplication.translate
        Record.setWindowTitle(_translate("Record", "Form"))
        self.FrameLabel.setText(_translate("Record", "TextLabel"))
        self.FrameLabel_2.setText(_translate("Record", "TextLabel"))
        self.label_4.setText(_translate("Record", "姓名"))
        self.label_2.setText(_translate("Record", "学号"))
        self.label_3.setText(_translate("Record", "姓名"))
        self.label.setText(_translate("Record", "学号"))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainDialog = QDialog()
    myDialog = Ui_Record()  # 创建对话框
    myDialog.setupUi(MainDialog)  # 将对话框依附于主窗体
    MainDialog.show()

    sys.exit(app.exec_())