# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from face_recognition import face_encodings, face_locations, face_landmarks



def eye_aspect_ratio(eye):
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


def eye_blink(frame, frame_count):
    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold
    EYE_AR_THRESH = 0.2
    EYE_AR_CONSEC_FRAMES = 1

    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    location = face_locations(frame, 1, 'cnn')
    marks = face_landmarks(frame, location, 'large')

    leftEye = marks[0]['left_eye']
    leftEye = np.array(leftEye)
    rightEye = marks[0]['right_eye']
    rightEye = np.array(rightEye)

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    print(ear)
    # compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    # check to see if the eye aspect ratio is below the blink
    # threshold, and if so, increment the blink frame counter
    if ear < EYE_AR_THRESH:
        COUNTER += 1

    # 突然变大，证明存在跳跃，即检测到眨眼
    else:
        # if the eyes were closed for a sufficient number of
        # then increment the total number of blinks
        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            TOTAL += 1
            frame_count['eye_blink_frames'] += 1

        # reset the eye frame counter
        COUNTER = 0


def mouth_aspect_ratio(top_lip, bottom_lip):
    A = dist.euclidean(top_lip[3], bottom_lip[9])
    C = dist.euclidean(top_lip[0], top_lip[6])
    ear = (2 * A) / (1.0 * C)
    return ear


def mouth_blink(frame, frame_count, COUNTER):
    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold
    MOUTH_AR_THRESH = 0.45
    MOUTH_AR_CONSEC_FRAMES = 1

    # initialize the frame counters and the total number of blinks
    #global COUNTER
    #COUNTER = 0
    TOTAL = 0

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    location = face_locations(frame, 1, 'cnn')
    marks = face_landmarks(frame, location, 'large')

    top_lip = marks[0]['top_lip']
    bottom_lip = marks[0]['bottom_lip']
    top_lip = np.array(top_lip)
    bottom_lip = np.array(bottom_lip)
    mouthEAR = mouth_aspect_ratio(top_lip, bottom_lip)
    print(mouthEAR)

    # compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    top_lipHull = cv2.convexHull(top_lip)
    bottom_lipHull = cv2.convexHull(bottom_lip)
    cv2.drawContours(frame, [top_lipHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [bottom_lipHull], -1, (0, 255, 0), 1)

    # check to see if the eye aspect ratio is below the blink
    # threshold, and if so, increment the blink frame counter
    if mouthEAR < MOUTH_AR_THRESH:
        COUNTER += 1
        print(COUNTER)

    # 突然变大，证明存在跳跃，即检测到眨眼
    else:
        # if the eyes were closed for a sufficient number of
        # then increment the total number of blinks
        if COUNTER >= MOUTH_AR_CONSEC_FRAMES:
            TOTAL += 1
            frame_count['eye_blink_frames'] += 1
            #frame_count['mouth_blink_frames'] += 1
        # reset the eye frame counter
        COUNTER = 0
    return COUNTER