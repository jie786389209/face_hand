from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
from face_recognition import face_locations, compare_faces, face_encodings, face_landmarks
from .util import put_chinese
import cv2
import logging
import numpy as np
#from face.MainWindows import
#import face.MainWindows.logQueue 存在互相导入问题
from imutils.video import VideoStream
import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
import imutils
import pickle
import time
import os
from .eye_blink import eye_blink, mouth_blink
import copy

# global COUNTER
# COUNTER = 0

class CVThead(QThread):
    '''多线程处理图像文件夹'''
    sin_out_names = pyqtSignal(list)
    sin_out_frame = pyqtSignal(np.ndarray)
    sin_out_pixmap = pyqtSignal(QPixmap)  # 自定义信号，执行run()函数时，从相关线程发射此信号
    fake_flag_signal = pyqtSignal(str)
    #str_signal = pyqtSignal(str) #通用信号
    #global cap_list

    def __init__(self):  # 初始化的各数据值
        super(CVThead, self).__init__()
        self.capID = 0
        self.cap_list = [cv2.VideoCapture(0), cv2.VideoCapture(1)]
        self.cap_used = None
        self.ratio = 0.25
        self.model = 'cnn'
        self.tolerance = 42
        self.up_sample = 1
        self.jitters = 10
        self.face = None
        self.face_num = 0
        self.encoding = None
        self.known_face_encodings = []
        self.known_face_names = []
        self.check_flag = False #相当于face.py的check_flag传过来改变了cvthread,py的check_flag
        self.fake_label = True
        self.liveness_enable = False
        self.frame_count = { 'total_frames':0, 'eye_blink_frames':0, 'liveness_detect_frames':0 } # 检查的帧数，默认检查5帧
        self.COUNTER = 0 # 计算动嘴的次数的
        self.mouth_enable = False

        # 初始化sess
        self.init_sess()



    def init_sess(self):
        # 初始化gpu的使用
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        global sess1
        sess1 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        global graph1
        graph1 = tf.get_default_graph()
        # graph1 = tf.Graph()

        # with graph.as_default():
        # load our serialized face detector from disk
        print("[INFO] loading face detector...")
        protoPath = os.path.sep.join(["./src/fake_face", "deploy.prototxt"])
        modelPath = os.path.sep.join(["./src/fake_face",
                                      "res10_300x300_ssd_iter_140000.caffemodel"])
        global net
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        # load the liveness detector model and label encoder from disk
        print("[INFO] loading liveness detector...")
        global model
        model = keras.models.load_model("./src/fake_face/jie_liveness.model")

        global le
        le = pickle.loads(open("./src/fake_face/jie_le.pickle", "rb").read())
        print("导入完成")


    def check(self):
        self.check_flag = True
        #print(self.Check_Flag)

    def __del__(self):
        self.wait()

    # def change_cap(self):
    #     self.cap_used = self.cap_list[self.capID]

    def work(self):
        #pass
        # 获取摄像图像进行人脸识别
        ret, origin_frame = self.cap_used.read()   # ret 子程序的返回指令

        if self.check_flag:
            #self.check_flag = False
            if not ret:
                logging.debug("cap not open")
                #logQueue.put('Error：初始化摄像头失败')
                return
            else:
                frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
                small_frame = cv2.resize(frame, (0, 0), fx=self.ratio, fy=self.ratio)
                # 活体检测
                if self.liveness_enable == True:
                    real_flag = self.live_detect(origin_frame)
                    if not real_flag: # 若为假人，不执行下面语句
                        self.frame_count['total_frames'] += 1
                        print(self.frame_count['total_frames'], self.frame_count['liveness_detect_frames'],
                              self.frame_count['eye_blink_frames'])
                        if self.frame_count['total_frames'] == 5:  # 统计这5帧的情况
                            # 一旦到达5帧就置零
                            self.frame_count['total_frames'] = 0
                            self.frame_count['eye_blink_frames'] = 0
                            self.frame_count['liveness_detect_frames'] = 0
                        return   # 这里返回，所以total_frame不能统一到下面加

                    else: # 若为真人
                        self.frame_count['total_frames'] += 1  # 计算帧数
                        self.frame_count['liveness_detect_frames'] += 1

                else:
                    pass
                    #sess1.close() 展示先不关闭

                # 眨眼检测
                # eye_blink(frame, self.frame_count)

                # 动嘴检测
                if self.mouth_enable == True:
                    # self.COUNTER = mouth_blink(frame, self.frame_count, copy.deepcopy(self.COUNTER))
                    self.COUNTER = mouth_blink(frame, self.frame_count, self.COUNTER)
                    if self.liveness_enable == False:
                        self.frame_count['total_frames'] += 1

                # 检测人脸
                locations = face_locations(small_frame,
                                           model=self.model,
                                           number_of_times_to_upsample=self.up_sample)   # 在face_recognition模块中调用人脸定位

                # 68个点
                # marks = face_landmarks(frame, locations, 'large')


                # 生成人脸向量
                encodings = face_encodings(small_frame,
                                           locations,
                                           num_jitters=self.jitters)

                names = []
                for encoding in encodings:
                    name = ""

                    matches = compare_faces(self.known_face_encodings,
                                            encoding,
                                            tolerance=self.tolerance/100)
                    # 如果匹配则使用第一个匹配的人名
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.known_face_names[first_match_index]

                    if name != "":
                        names.append(name)


                # 显示人脸位置与人名标注
                for (top, right, bottom, left), name in zip(locations, names):
                    # 放大回原图大小
                    top, right, bottom, left = (x*4 for x in (top, right, bottom, left))
                    # 显示方框
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    if name != "":
                        cv2.rectangle(frame, (left, bottom + 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        # 显示人名
                        frame = put_chinese(frame, (left + 10, bottom), name, (255, 255, 255), 25)
                # 转换成QImage
                qformat = QImage.Format_RGB888
                out_image = QImage(frame,
                                   frame.shape[1],
                                   frame.shape[0],
                                   frame.strides[0],
                                   qformat)
                print(self.frame_count['total_frames'], self.frame_count['liveness_detect_frames'], self.frame_count['eye_blink_frames'])
                #self.frame_count['total_frames'] = self.frame_count['total_frames'] + 1  # 计算帧数
                if self.frame_count['total_frames'] == 5: #统计这5帧的情况
                    if self.frame_count['eye_blink_frames'] >= 1 or self.frame_count['liveness_detect_frames'] >= 3:
                        self.sin_out_names.emit(names)


                    # 一旦到达5帧就置零
                    self.frame_count['total_frames'] = 0
                    self.frame_count['eye_blink_frames'] = 0
                    self.frame_count['liveness_detect_frames'] = 0



                # 输出信号
                #self.sin_out_names.emit(names)
                self.sin_out_frame.emit(origin_frame.copy())
                self.sin_out_pixmap.emit(QPixmap.fromImage(out_image).copy())  # 传出副本，不然会出现GBR

        else:
            if self.cap_list[0].isOpened() or self.cap_list[1].isOpened():
                frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
                self.sin_out_frame.emit(origin_frame.copy())
                qformat = QImage.Format_RGB888
                out_image = QImage(frame,
                                   frame.shape[1],
                                   frame.shape[0],
                                   frame.strides[0],
                                   qformat)

                self.sin_out_pixmap.emit(QPixmap.fromImage(out_image).copy())  # 传出副本，不然会出现GBR




    def run(self):
        self.cap_used = self.cap_list[self.capID]
        frames_count = 0
        while True:
            if self.check_flag:
                frames_count +=1
                if frames_count != 5:
                    self.work()
                else:
                    frames_count = 0
                    self.check_flag = False
            else:
                self.work()


    def live_detect(self, frame):
        fake_ornot_frame = imutils.resize(frame, width=600)
        (h, w) = fake_ornot_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(fake_ornot_frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the detected bounding box does fall outside the
                # dimensions of the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # extract the face ROI and then preproces it in the exact
                # same manner as our training data
                face = fake_ornot_frame[startY:endY, startX:endX]
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = tf.keras.preprocessing.image.img_to_array(face)
                face = np.expand_dims(face, axis=0)

                # pass the face ROI through the trained liveness detector
                # model to determine if the face is "real" or "fake"
                with sess1.as_default():
                    with graph1.as_default():
                        sess1.run(tf.global_variables_initializer())
                        preds = model.predict(face)[0]
                # CVThead.sess.close()

                j = np.argmax(preds)
                self.fake_label = le.classes_[j]
                print(self.fake_label)
                if self.fake_label == 'fake':
                    self.fake_flag_signal.emit(self.fake_label + '：不是个活人')
                    return False
                else:
                    self.fake_flag_signal.emit(self.fake_label + '：是个活人')

                # draw the label and bounding box on the frame
                label = "{}: {:.4f}".format(self.fake_label, preds[j])
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                break  # 只保留一个label
        return True






