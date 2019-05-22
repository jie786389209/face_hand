from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog
import os
#from .util import get_encodings
from face_recognition import load_image_file, face_encodings
from PyQt5 import QtWidgets


class BatchRecordThead(QThread, QtWidgets.QWidget):
    sin_out_tuple = pyqtSignal(tuple)
    t_c_list = pyqtSignal(list) #定义信号teacher和classroom列表
    log_batch = pyqtSignal(str) #log发射的信号

    def __init__(self, batch_path):
        super(BatchRecordThead, self).__init__()
        self.batch_path = batch_path
        #self.file_nums = 0

    def run(self):

        if self.batch_path == '':
            return

        teachers_path = self.batch_path.split('/') ##分割路径
        teacher_classroom = teachers_path[-1] ##获取文件名
        t_c = teacher_classroom.split('_')##分割老师和教师
        t = t_c[0]
        c = t_c[-1]

        self.t_c_list.emit([t, c]) #发射获得的teacher和student的信息到主模块

        known_face_names, known_face_encodings = self.get_encodings(self.batch_path)

        self.sin_out_tuple.emit((known_face_names, known_face_encodings))

    def get_encodings(self, path, jitters=100):
        files = os.listdir(path)
        known_face_names = []
        known_face_encodings = []

        #self.pbar = QProgressBar()

        for file in files:
            # self.file_nums = self.file_nums + 1
            # self.pbar.setValue(self.file_nums)
            # self.pbar.show()

            if file.endswith('.jpg') or file.endswith('.png'):
                name = file.split('.')[0]
                known_face_names.append(name)
                img = load_image_file(path + '/' + file)

                try:
                    encodings = face_encodings(img, num_jitters=jitters)[0]
                    known_face_encodings.append(encodings)
                    self.log_batch.emit(file + '识别成功')
                except Exception as e:
                    print(file, '无法识别')
                    self.log_batch.emit(file + '识别失败')
        print("所有图片都识别成功")
        self.log_batch.emit(file + '所有图片都识别成功')
        return known_face_names, known_face_encodings




