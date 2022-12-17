import cv2
import numpy as np
import time
import easyocr
from paddleocr import PaddleOCR
import re
from itertools import groupby
from helpers.infer import pre_process, post_process
from helpers.ocr import extract_result
from urllib.request import urlopen

CAMERA_BUFFER_SIZE = 3  # buffrer frames to drop for webcam
SKIP_FRAMES_ONSUCCESS = 50  # 0-50 after found number skip relax for a few frames
UNFOUND_PLATE_STRING = "XXXXXXXX"  # default plate nr if problem detecting frame
# accept ocr only same result received on so many frames (5-8) readings repeatedly
TIMES_CANDIDATES_REPEATED_TO_ACCEPT = 8
# if only one number detected on image be specially sure
SINGLE_DETECT_CONFIDENCE_TO_PASS = 0.86
CAMERA_ADDRESS = "rtsp://admin:AnafigA_123@192.168.20.194:554/media/video1"
CAMERA_NAME = "22(194)"
PAUSE_ON_ERROR_IN_STREAM = 10
NUMBER_OF_TRIALS_TO_RESTORE_STREAM = 20000


def generateStrinsList(count):
    res = []
    for i in range(1, count + 1):
        res.append(str(i))
    return res


def downsize_frame(frame_input, max_height=1080):
    if frame_input.shape[0] > 1081:
        height = max_height
        width = int(frame_input.shape[1] * height / frame_input.shape[0])
        dim = (width, height)
        frame_input = cv2.resize(
            frame_input, dim, interpolation=cv2.INTER_AREA)
    return frame_input


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def inferFrame(frame, net):
    detections = pre_process(frame, net)
    return post_process(frame.copy(), detections)


def chooseBest(results, confids):
    def_result = UNFOUND_PLATE_STRING
    results = [re.sub("[^0-9]", "", ss) for ss in results]
    if len(results) == 1 and len(results[0]) == 8:
        # print("single detect", results[0], confids[0])  # DEBUG
        if (confids[0] > SINGLE_DETECT_CONFIDENCE_TO_PASS):
            return results[0]
        else:
            return def_result
    if len(results) == 0:
        return def_result
    max_conf_value = max(confids)
    max_conf_index = confids.index(max_conf_value)
    highest_conf_result = results[max_conf_index]
    # print(results)  # DEBUG
    # print(confids)  # DEBUG
    # print("passing to candidates",
    #       highest_conf_result,
    #       len(highest_conf_result)
    #       )  # DEBUG
    if len(highest_conf_result) != 8:
        return def_result
    else:
        return highest_conf_result


def processDetectionInImage(img, reader, pocr):
    return extract_result(img, reader, pocr)
    # print(extract_result(img, reader))


def processFrame(frame, net, reader, pocr):
    frame = downsize_frame(frame)
    imgs = inferFrame(frame, net)
    results = []
    confids = []
    plate = UNFOUND_PLATE_STRING
    for img in imgs:
        found, conf = processDetectionInImage(img, reader, pocr)
        if found != "":
            results.append(found)
            confids.append(conf)
    if len(results) > 0:
        plate = chooseBest(results, confids)
    # print(plate)
    # if plate != UNFOUND_PLATE_STRING:
    #     print("-------detected----------")
    return plate


def processStream(file_name, net, reader, pocr):
    cap = cv2.VideoCapture(file_name)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    # Read until video is completed
    latest_detection = UNFOUND_PLATE_STRING
    candidates = generateStrinsList(TIMES_CANDIDATES_REPEATED_TO_ACCEPT)
    # print("CANDIDATES", candidates)  # DEBUG
    skip_frames_remaning = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if skip_frames_remaning > 0:
            skip_frames_remaning -= 1
            continue
        if ret == True:
            candidate = processFrame(frame, net, reader, pocr)
            if candidate != UNFOUND_PLATE_STRING:
                candidates.insert(0, candidate)
                candidates.pop()
                # print(candidates)  # DEBUG
                if all_equal(candidates) and candidates[0] != UNFOUND_PLATE_STRING and candidates[0] != latest_detection:
                    latest_detection = candidates[0]
                    if len(latest_detection) == 8:
                        print("===============================================")
                        print(
                            f"====== Found good detection    {latest_detection}     - congrats  ===")
                        print("===============================================")
                        try:
                            urlopen(  # nosec
                                f'http://notscr.amgs.me/info/ocr.aspx?nr={latest_detection}&camera={CAMERA_NAME}', timeout=0.5)
                        except Exception as e:
                            print("Caannot open amgs, skipping")
                        skip_frames_remaning = SKIP_FRAMES_ONSUCCESS
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        # Break the loop if ret not true
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frame = cv2.imread('test.jpg')
    net = cv2.dnn.readNet('../models/best_simp.onnx')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    reader = easyocr.Reader(['en'])
    pocr = PaddleOCR(use_angle_cls=True, lang='en',
                     debug=False, show_log=False)
    start = time.time()
    # processFrame(frame, net, reader, pocr)
    video_file_name = "01.ts"
    video_strem_name = CAMERA_ADDRESS
    for trial in range(1, NUMBER_OF_TRIALS_TO_RESTORE_STREAM):
        processStream(video_strem_name, net, reader, pocr)
        print(
            f"Trial {trial}. No stream received, pausing for PAUSE_ON_ERROR_IN_STREAM: {PAUSE_ON_ERROR_IN_STREAM} seconds")
        time.sleep(PAUSE_ON_ERROR_IN_STREAM)

    # print(chooseBest([]))
    end = time.time()
    print("[INFO] Frame detection function took {:.6f} s".format(
        end - start))
