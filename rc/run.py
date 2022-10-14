import cv2
import numpy as np
import time
import easyocr
from helpers.infer import pre_process, post_process
from helpers.ocr import extract_result


def inferFrame(frame, net):
    detections = pre_process(frame, net)
    return post_process(frame.copy(), detections)


def processStream():
    net = cv2.dnn.readNet('../models/best_simp.onnx')
    reader = easyocr.Reader(['en'])
    cap = cv2.VideoCapture('01.ts')
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            start = time.time()
            inferFrame(frame, net)
            end = time.time()
            print("[INFO] Frame detection function took {:.6f} s".format(
                end - start))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()


def processFrame(frame, net, reader):
    inferFrame(frame, net)
    print(extract_result(img, reader))


if __name__ == '__main__':
    frame = cv2.imread('test.jpg')
    net = cv2.dnn.readNet('../models/best_simp.onnx')
    reader = easyocr.Reader(['en'])
    imgs = inferFrame(frame, net)
    print("Start frame ocr")
    start = time.time()
    for img in imgs:
        processFrame(frame, net, reader)
    end = time.time()
    print("[INFO] Frame detection function took {:.6f} s".format(
        end - start))
