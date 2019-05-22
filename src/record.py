from PyQt5.QtGui import QImage,QRegExpValidator, QPixmap
from PyQt5.QtCore import QRegExp
from PyQt5.QtWidgets import *
from .recordUI import Ui_Record
from face_recognition import face_encodings, face_locations, face_landmarks
import cv2
from scipy.spatial import distance as dist
import numpy as np

class RecordForm(QDialog, Ui_Record):
    def __init__(self, frame):
        super(RecordForm, self).__init__()
        self.setupUi(self)
        self.location = None
        self.encoding = None
        self.name = None
        self.number = None
        self.face_img = None
        self.frame = frame

        # 人脸识别
        try:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.location = face_locations(self.frame, 1, 'cnn')
            if len(self.location) == 1:
                self.encoding = face_encodings(self.frame, self.location, 10)
                self.marks = face_landmarks(self.frame, self.location, 'large')
                # 画眼眶
                self.draw_mouth()
                #self.draw_eye()

                # 标记关键点
                for person in self.marks:
                    for points, positions in person.items():
                        for position in positions:
                            cv2.circle(self.frame, position, 2, (0, 255, 0), thickness=2)
                top, right, bottom, left = self.location[0]
                self.face_img = self.frame[top-35:bottom+35, left-35:right+35, :]
        except IndexError:
            QMessageBox.question(self, 'Warning', "未检测到关键点",
                                 QMessageBox.Yes, QMessageBox.Yes)

        # 显示图片
        qformat = QImage.Format_RGB888
        if self.face_img is None:
            self.close()
        else:
            self.face_img = cv2.resize(self.face_img,
                                       (self.FrameLabel.height(), self.FrameLabel.width()))
            out_image = QImage(self.face_img,
                               self.face_img.shape[1],
                               self.face_img.shape[0],
                               self.face_img.strides[0],
                               qformat)
            self.FrameLabel.setPixmap(QPixmap.fromImage(out_image))
            self.FrameLabel.setScaledContents(True)

        # 正则表达式限制输入
        name_regx = QRegExp('^[\u4e00-\u9fa5]{1,10}$')
        name_validator = QRegExpValidator(name_regx, self.NameLineEdit)
        self.NameLineEdit.setValidator(name_validator)
        self.NameLineEdit.setText("请输入中文名")

        # # 学号输入
        # number_regx = QRegExp('^[\u4e00-\u9fa5]{1,10}$')
        # number_validator = QRegExpValidator(number_regx, self.LineEdit)
        # self.LineEdit.setValidator(number_validator)
        # self.LineEdit.setText("请输入学号")

        # 判断是否保存注册
        self.DialogBox.accepted.connect(self.dialog_box_accept)
        self.DialogBox.rejected.connect(self.dialog_box_reject)

    def dialog_box_accept(self):
        self.name = self.NameLineEdit.text()
        # self.number = self.LineEdit.text()
        self.close()

    def dialog_box_reject(self):
        self.close()

    def closeEvent(self, event):
        self.close()

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear

    def mouth_aspect_ratio(self, top_lip, bottom_lip):
        A = dist.euclidean(top_lip[3], bottom_lip[9])
        C = dist.euclidean(top_lip[0], top_lip[6])
        ear = A / (1.0 * C)
        return ear

    def draw_eye(self):
        leftEye = self.marks[0]['left_eye']
        leftEye = np.array(leftEye)
        rightEye = self.marks[0]['right_eye']
        rightEye = np.array(rightEye)

        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        print(ear)
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(self.frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(self.frame, [rightEyeHull], -1, (0, 255, 0), 1)

    def draw_mouth(self):
        top_lip = self.marks[0]['top_lip']
        bottom_lip = self.marks[0]['bottom_lip']
        top_lip = np.array(top_lip)
        bottom_lip = np.array(bottom_lip)
        mouthEAR = self.mouth_aspect_ratio(top_lip, bottom_lip)
        print(mouthEAR)

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        top_lipHull = cv2.convexHull(top_lip)
        bottom_lipHull = cv2.convexHull(bottom_lip)
        cv2.drawContours(self.frame, [top_lipHull], -1, (0, 255, 0), 1)
        cv2.drawContours(self.frame, [bottom_lipHull], -1, (0, 255, 0), 1)

