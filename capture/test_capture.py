import cv2
import numpy as np
import time
from wurlitzer import pipes


CAMERA_ADDRESS = "rtsp://admin:AnafigA_123@192.168.20.223:554/media/video1"
CAMERA_BUFFER_SIZE = 6  # buffrer frames to drop for webcam
PAUSE_ON_ERROR_IN_STREAM = 2


def processStream(file_name):
    cap = cv2.VideoCapture(file_name)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # print("...Going.........................")


if __name__ == '__main__':
    for trial in range(1, 2):
        video_strem_name = CAMERA_ADDRESS
        with pipes() as (out, err):
            processStream(video_strem_name)
        print(
            f"Trial {trial}. No stream received, pausing for PAUSE_ON_ERROR_IN_STREAM: {PAUSE_ON_ERROR_IN_STREAM} seconds")
        time.sleep(PAUSE_ON_ERROR_IN_STREAM)
    print(out.read())
