import numpy as np
from PyQt5.QtCore import Qt, QTimer, QDateTime, pyqtSignal, QObject
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QTextCursor
from PyQt5.uic import loadUi
from src.MainWindow import Ui_MainWindow
from src.batchRecord import BatchRecordThead
from src.record import RecordForm
from src.data_Record import DataRecordUI
from src.data_Manage import DataManageUI
import pickle
import tensorflow as tf
from tensorflow import keras
import sys
import os
import cv2

import logging
import logging.config
import sqlite3
import threading
import queue
import multiprocessing
from datetime import datetime

from src.cvthread import CVThead #关键导入CVThead
from src.log import log_class



# logQueue = multiprocessing.Queue() #日志队列
# receiveLogSignal = pyqtSignal(str) #log信号

# # 日志系统
# receiveLogSignal.connect(lambda log: logOutput(log))
# logOutputThread = threading.Thread(target=receiveLog, daemon=True) # 设置为daemon，槽函数里面为while循环
# logOutputThread.start()
#
#
# # LOG输出
# def logOutput(log):
#     time = datetime.now().strftime('[%Y/%m/%d %H:%M:%S]')
#     log = time + ' ' + log + '\n'
#     MainWindows.log_box.moveCursor(QTextCursor.End)
#     MainWindows.log_box.insertPlainText(log)
#     MainWindows.log_box.ensureCursorVisible()  # 自动滚屏
#
# # 日志队列一旦有消息，就立刻发送到相应的槽里面（执行显示）
# def receiveLog():
#     while True:
#         data = logQueue.get()
#         if data:
#             receiveLogSignal.emit(data)
#         else:
#             continue



class MainWindows(QMainWindow, Ui_MainWindow):  # 自己建的子类MainWindows，继承父类QMainWindows和父类UI-MainWindows

    sin_out_flag = pyqtSignal(bool) #签到信号

    #
    # logQueue = multiprocessing.Queue() #日志队列
    # receiveLogSignal = pyqtSignal(str) #log信号

    def __init__(self):                       # 以下为子类
        super(MainWindows, self).__init__()  # super关键字，将子类MainWindows的对象self转换为QMainWindows的对象，被转换的self调用自己的函数-init-函数
        #self.setupUi(self)  # 直接继承界面类，调用类的setupUi方法
        loadUi('./src/MainWindow.ui', self)
        # self.setFixedSize(self.width(), self.height())

        self.frame = None
        self.pixmap = None
        self.cap_open_flag = False
        self.people_sum = 0
        self.people_attendence = []
        self.people_absence = []
        self.teacher = []
        self.classroom = []
        self.batch_path = ''
        # self.offList = []  #缺勤列表
        # self.totalPeople = [] #总人数列表

        # 显示当前时间
        self.dateTimeEdit.setDateTime(QDateTime.currentDateTime())  # 显示当前时间

        # 开启log
        self.log_msg = log_class()
        self.log_msg.receiveLogSignal.connect(lambda log: self.logOutput(log)) #CVThead对应的信号槽
        #self.log_msg.log_batch.connect(lambda log: self.logOutput(log)) #BatchThead对应的信号槽

        # 定时器
        # self.timer = QTimer(self)   # 实例化一个timer
        # self.timer.timeout.connect(self.update_frame)   # 设置定时执行的函数update—frame

        # OpenCV 线程
        self.cv_thread = CVThead()
        self.cv_thread.sin_out_pixmap.connect(self.process_signal)
        self.cv_thread.sin_out_frame.connect(self.process_signal)
        self.cv_thread.sin_out_names.connect(self.process_signal)
        # 活体信号
        self.cv_thread.fake_flag_signal.connect(lambda log: self.logOutput(log))

        # 手动签到
        self.check_flag = False
        self.push_attend.clicked.connect(self.cv_thread.check) #发送到cv_thread的check函数

        # self.radioButtonCNN.setChecked(True)
        # self.radioButtonCNN.toggled.connect(self.change_model)
        # self.radioButtonHOG.setChecked(False)
        # self.radioButtonHOG.toggled.connect(self.change_model)

        # 活体检测
        self.live_detectButton.setChecked(False)
        self.live_detectButton.toggled.connect(self.liveDetect_enable)
        # 动嘴检测
        self.mouth_detectButton.setChecked(False)
        self.mouth_detectButton.toggled.connect(self.mouthDetect_enable)

        # 容错率
        self.valueLabel.setText(str(self.cv_thread.tolerance))  # 设置按钮的显示文本
        self.toleranceSlider.setMinimum(0)  # 设置容错率的最大值与最小值
        self.toleranceSlider.setMaximum(100)
        self.toleranceSlider.setSingleStep(1)
        self.toleranceSlider.setValue(self.cv_thread.tolerance)
        self.toleranceSlider.valueChanged.connect(self.tolerance_state)

        # 现场注册
        self.registerButton.setEnabled(False)  # setEnabled设置按钮是否可以使用，如果为False则不可以用，点击也不会发射信号
        self.registerButton.clicked.connect(self.record)    # 批量注册

        # db数据库
        self.dataRecord_button.clicked.connect(self.db_record)
        self.dataManage_button.clicked.connect(self.db_manage)

        # 后台批量注册
        self.batchRecordThead = BatchRecordThead(self.batch_path)
        self.batchRecordThead.log_batch.connect(lambda log: self.logOutput(log))
        # 批量注册路径下的老师和学生信息，显示到屏幕上
        self.batchRecordThead.t_c_list.connect(self.tc_process)
        # 批量注册返回的name和encoding，将其添加到全局名中
        self.batchRecordThead.sin_out_tuple.connect(self.batch_record)
        self.batchRegisterButton.clicked.connect(self.batch_record)

        # 导入导出成pkl文件，后期改成数据库
        self.loadInButton.clicked.connect(self.load_in)
        self.loadOutButton.clicked.connect(self.load_out)

        # 相机设置
        self.CameraCheckBox.stateChanged.connect(self.use_external_camera)
        self.CameraButton.toggled.connect(self.start_camera)
        self.CameraButton.setCheckable(True)


    def load_out(self):
        with open('encodings.pkl', 'wb') as f:
            pickle.dump(self.cv_thread.known_face_encodings, f)
        with open('names.pkl', 'wb') as f:
            pickle.dump(self.cv_thread.known_face_names, f)
        self.log_msg.logQueue.put('成功导出pkl文件')

    def load_in(self):
        try:
            with open('encodings.pkl', 'rb') as f:
                self.cv_thread.known_face_encodings = pickle.load(f)
            with open('names.pkl', 'rb') as f:
                self.cv_thread.known_face_names = pickle.load(f)
        except IOError:
            print("Files not found")
            self.log_msg.logQueue.put('导入pkl文件失败')

    def change_model(self):
        pass

    # 读取路径下的老师和学生信息
    def tc_process(self, signal):
        self.teacherEdit.clear()
        self.classEdit.clear()
        self.teacher = signal[0]
        self.classroom = signal[-1]
        self.teacherEdit.setText(self.teacher)
        self.classEdit.setText(self.classroom)


    # 处理信号
    def process_signal(self, signal):
        if isinstance(signal, QPixmap):
            self.pixmap = signal
        elif isinstance(signal, list):
            if self.cap_open_flag:
                if len(signal) == 0:
                    pass
                elif signal not in self.people_attendence:
                    self.people_attendence.extend(signal)
                else:
                    pass
                self.people_sum = len(self.people_attendence)  # 赋值 签到总人数
                self.attendance_people.setTitle("签到人数：{0}".format(self.people_sum))  # 计算出签到人数

                #set1 = set(self.cv_thread.known_face_names)
                #set2 = set(self.people_attendence)

                self.people_absence = list(set(self.cv_thread.known_face_names) - set(self.people_attendence))
                self.attendance_people_text.setText('\n'.join('%s' % id for id in self.people_attendence))  # 显示签到人名
                self.absence_people_text.setText('\n'.join('%s' % id for id in self.people_absence))



        elif isinstance(signal, np.ndarray):
            self.frame = signal

    # 是否使用外接摄像头
    def use_external_camera(self, status):
        if status == Qt.Checked:
            self.cv_thread.capID = 1
            self.cv_thread.change_cap()
        else:
            self.cv_thread.capID = 0
            self.cv_thread.change_cap()

    # 打开/关闭摄像头
    def start_camera(self, status):
        if status:
            self.timer = QTimer(self)  # 实例化一个timer
            self.timer.timeout.connect(self.update_frame)  # 设置定时执行的函数update—frame
            self.cap_open_flag = True
            self.cv_thread.start()
            self.log_msg.logQueue.put('成功开启CVThread线程采集人脸')
            self.CameraButton.setText('关闭摄像头')
            self.registerButton.setEnabled(True)  # 按钮变为可切换状态
            self.timer.start()
        else:
            self.cap_open_flag = False
            self.cv_thread.quit()
            self.log_msg.logQueue.put('成功关闭CVThread线程采集人脸')
            if self.timer.isActive():
                self.timer.stop()
            self.CameraButton.setText('打开摄像头')
            self.registerButton.setEnabled(False)
            self.FrameLabel.clear()  # 清屏 并且左侧显示当前人员
            self.groupBox.setTitle("当前人员")
            #self.nameBrowser.clear()

    # 更新画面
    def update_frame(self):
        if self.cap_open_flag and self.pixmap is not None:
            self.FrameLabel.setPixmap(self.pixmap)
            self.FrameLabel.setScaledContents(True)
        else:
            self.FrameLabel.clear()

    # 修改tolerance
    def tolerance_state(self):
        self.cv_thread.tolerance = self.toleranceSlider.value()
        self.valueLabel.setText(str(self.cv_thread.tolerance))

    # 用户注册
    def record(self):
        self.log_msg.logQueue.put('请将人脸靠近摄像头')
        QMessageBox.question(self, 'Information', "请将人脸靠近摄像头", QMessageBox.Yes, QMessageBox.Yes)
        # 用户信息填写对话框
        if self.frame is not None:
            dialog_window = RecordForm(self.frame.copy())
            dialog_window.exec_()
            if dialog_window.name is not None:
                self.cv_thread.known_face_encodings.extend(dialog_window.encoding)
                self.cv_thread.known_face_names.append(dialog_window.name)
                QMessageBox.question(self, 'Information', "注册成功", QMessageBox.Yes, QMessageBox.Yes)
            del dialog_window
        else:
            QMessageBox.question(self, 'Warning', "未检测到人脸", QMessageBox.Yes, QMessageBox.Yes)

    # 通过照片路径注册
    def batch_record(self, signal):
        if not signal:
            self.batch_path = QFileDialog.getExistingDirectory(self, "选取文件夹", "./",
                                                               options=QFileDialog.DontUseNativeDialog)  # 注意不要在线程里面使用GUI，容易crash，所以batchThread中把读入路径放到这里
            self.batchRecordThead.batch_path = self.batch_path
        if isinstance(signal, tuple):
            # self.totalPeople.append(signal)
            self.cv_thread.known_face_names.extend(signal[0])
            self.cv_thread.known_face_encodings.extend(signal[1])
            self.batchRecordThead.quit()
        else:
            self.batchRecordThead.start()
            self.log_msg.logQueue.put('开启batchRecordThead线程')

    def save_data(self, list1, list2):
        self.known_face_encodings = list2
        self.known_face_names = list1

    # LOG输出
    def logOutput(self, log):
        from face import MainWindows  # 在这里导入，如果一开始就导入会出现问题
        if ('fake' or 'real') in log:
            QMessageBox.question(self, 'Warning', "请使用活体来签到", QMessageBox.Yes, QMessageBox.Yes)
        time = datetime.now().strftime('[%Y/%m/%d %H:%M:%S]')
        log = time + ' ' + log + '\n'
        self.log_box.moveCursor(QTextCursor.End)
        self.log_box.insertPlainText(log)
        self.log_box.ensureCursorVisible()  # 自动滚屏

    def liveDetect_enable(self):
        if self.live_detectButton.isChecked():
            self.cv_thread.liveness_enable = True
        else:
            self.cv_thread.liveness_enable = False

    def mouthDetect_enable(self):
        if self.mouth_detectButton.isChecked():
            self.cv_thread.mouth_enable = True
        else:
            self.cv_thread.mouth_enable = False

    # 数据库
    def db_record(self):
        if self.cap_open_flag==True:
            self.start_camera(status=0)
        #self.cv_thread.quit()
        if self.cv_thread.cap_list[0].isOpened():
            self.cv_thread.cap_list[0].release()
        elif self.cv_thread.cap_list[1].isOpened():
            self.cv_thread.cap_list[1].release()
        #cv2.destroyAllWindows()


        dbRecord_window = DataRecordUI()
        dbRecord_window.exec_()
        # if db_window.name is not None:
        #     QMessageBox.question(self, 'Information', "数据库操作成功", QMessageBox.Yes, QMessageBox.Yes)
        del dbRecord_window

        # if not self.cv_thread.cap_list[0].isOpened():
        #     self.cv_thread.cap_used = self.cv_thread.cap_list[0]
        # elif not self.cv_thread.cap_list[1].isOpened():
        #     self.cv_thread.cap_used = self.cv_thread.cap_list[1]
        # self.start_camera(status=1)

        # OpenCV 线程
        self.cv_thread = CVThead()
        self.cv_thread.sin_out_pixmap.connect(self.process_signal)
        self.cv_thread.sin_out_frame.connect(self.process_signal)
        self.cv_thread.sin_out_names.connect(self.process_signal)

        # 活体信号
        self.cv_thread.fake_flag_signal.connect(lambda log: self.logOutput(log))


        # 手动签到
        self.check_flag = False
        self.push_attend.clicked.connect(self.cv_thread.check)  # 发送到cv_thread的check函数

        # 容错率
        self.valueLabel.setText(str(self.cv_thread.tolerance))  # 设置按钮的显示文本
        self.toleranceSlider.setMinimum(0)  # 设置容错率的最大值与最小值
        self.toleranceSlider.setMaximum(100)
        self.toleranceSlider.setSingleStep(1)
        self.toleranceSlider.setValue(self.cv_thread.tolerance)
        self.toleranceSlider.valueChanged.connect(self.tolerance_state)

    def db_manage(self):
        dbManage_window = DataManageUI()
        dbManage_window.exec_()
        del dbManage_window


if __name__ == "__main__":
    app = QApplication(sys.argv)
    #global window
    window = MainWindows()
    window.show()
    sys.exit(app.exec_())




