import cv2
import numpy as np
import time
import easyocr
import re
from itertools import groupby
from helpers.infer import pre_process, post_process
from helpers.ocr import extract_result

CAMERA_BUFFER_SIZE = 3


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def inferFrame(frame, net):
    detections = pre_process(frame, net)
    return post_process(frame.copy(), detections)


def chooseBest(results, confids):
    results = [re.sub("[^0-9]", "", ss) for ss in results]
    if len(results) == 1 and len(results[0]) == 8:
        return results[0]
    def_result = "XXXXXXXX"
    if len(results) == 0:
        return def_result
    max_conf_value = max(confids)
    max_conf_index = confids.index(max_conf_value)
    highest_conf_result = results[max_conf_index]
    if len(highest_conf_result) != 8:
        return def_result
    else:
        return highest_conf_result


def processDetectionInImage(img, reader):
    return extract_result(img, reader)
    # print(extract_result(img, reader))


def processFrame(frame, net, reader):
    imgs = inferFrame(frame, net)
    results = []
    confids = []
    for img in imgs:
        found, conf = processDetectionInImage(img, reader)
        if found != "":
            results.append(found)
            confids.append(conf)
    plate = chooseBest(results, confids)
    # print(plate)
    # if plate != "XXXXXXXX":
    #     print("-------detected----------")
    return plate


def processStream(net, reader):
    cap = cv2.VideoCapture('03.mp4')
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    # Read until video is completed
    latest_detection = "XXXXXXXX"
    candidates = ["a", "b", "c", "d", "e", "f"]
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            candidate = processFrame(frame, net, reader)
            if candidate != "XXXXXXXX":
                candidates.insert(0, candidate)
                candidates.pop()
                # print(candidates)
                if all_equal(candidates) and candidates[0] != "XXXXXXXX" and candidates[0] != latest_detection:
                    latest_detection = candidates[0]
                    if len(latest_detection) == 8:
                        print(
                            f"Found good detection {latest_detection} - congrats")
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frame = cv2.imread('test.jpg')
    net = cv2.dnn.readNet('../models/best_simp.onnx')
    reader = easyocr.Reader(['en'])
    start = time.time()
    # processFrame(frame, net, reader)
    processStream(net, reader)
    # print(chooseBest([]))
    end = time.time()
    print("[INFO] Frame detection function took {:.6f} s".format(
        end - start))
