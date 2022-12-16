import cv2
import numpy as np
import time
from wurlitzer import pipes
from helpers.video_threading import VideoGet, VideoShow


CAMERA_ADDRESS = "rtsp://admin:AnafigA_123@192.168.20.194:554/media/video1"
CAMERA_BUFFER_SIZE = 6  # buffrer frames to drop for webcam
PAUSE_ON_ERROR_IN_STREAM = 3


def processStream(file_name):
    video_getter = VideoGet(CAMERA_ADDRESS).start()
    while video_getter.stream.isOpened():
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break
        frame = video_getter.frame
        cv2.imshow("Main window", frame)
        grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("BW window", grayImage)

    # cap = cv2.VideoCapture(file_name)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    # if (cap.isOpened() == False):
    #     print("Error opening video stream or file")
    # while (cap.isOpened()):
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #     if ret:
    #         cv2.imshow("Main window", frame)
    #         grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         cv2.imshow("BW window", grayImage)
    #     # print("...Going.........................")


if __name__ == '__main__':
    for trial in range(1, 3):
        video_strem_name = CAMERA_ADDRESS
        processStream(video_strem_name)
        # with pipes() as (out, err):
        #     processStream(video_strem_name)
        # print(
        #     f"Trial {trial}. No stream received, pausing for PAUSE_ON_ERROR_IN_STREAM: {PAUSE_ON_ERROR_IN_STREAM} seconds")
        time.sleep(PAUSE_ON_ERROR_IN_STREAM)
    # print(out.read())
