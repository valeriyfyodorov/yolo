# import keras_ocr
import pytesseract
import cv2
import easyocr
import time
import os
import numpy as np

EASY_MIN_CONFIDENCE_LEVEL = 0.55


def processText(txt):
    false_flags = [",", '"', " ", "!", "\n", "'"]
    for flag in false_flags:
        txt = txt.replace(flag, "")
    if len(txt) > 8:
        txt = txt[:8]
    return txt


def extract_tesser(image):
    text = pytesseract.image_to_string(image)
    return processText(text)


def extract_easy(reader, image):
    box = reader.readtext(image)
    conf = 0
    res = ""
    if len(box) > 0:
        res = processText(box[0][1])
        if len(box[0]) > 2:
            conf = box[0][2]
    return res, conf


def processImage(image, resize=False):
    if image.shape[1] < 80 or image.shape[1] > 300 or resize:
        new_width = 256
        new_height = int(new_width / image.shape[1] * image.shape[0])
        image = cv2.resize(image,
                           (new_width, new_height),
                           interpolation=cv2.INTER_CUBIC)
    return image


def jpgsIntoList(dir_path):
    res = []
    # Iterate directory
    for file in os.listdir(dir_path):
        if file.endswith('.jpg'):
            res.append(file)
    return res


def run_tests(file_names):
    reader = easyocr.Reader(['en'])
    start = time.time()
    print("Looping tesser")
    start = time.time()
    for file in file_names:
        image = cv2.imread(file)
        extract_tesser(processImage(image), file)
    print("[INFO] Tesser took {:.6f} seconds".format(time.time() - start))
    print("Looping easy")
    start = time.time()
    for file in file_names:
        image = cv2.imread(file)
        extract_easy(reader, processImage(image, resize=True), file)
    print("[INFO] Easy took {:.6f} seconds".format(time.time() - start))


def extract_result(image, reader):
    width = image.shape[1]
    if image.shape[0] < 10 or width < 20:
        return "", 0
    tesser_confid = 0.4
    if width < 100:
        tesser_confid = 0.3
    # print(width)
    if width > 150:
        result, confid = extract_easy(reader, processImage(image, resize=True))
    else:
        result, confid = "", 0.2
    # print(result, confid)
    # try tesser where easy confidence is low
    if confid < EASY_MIN_CONFIDENCE_LEVEL or len(result) < 8:
        text_tesser = extract_tesser(processImage(image))
        if len(text_tesser) >= 8:
            result = text_tesser
            confid = tesser_confid
            # print(result, confid, "----tesser---")
    if len(result) < 8:
        result = ""
    return result, confid


if __name__ == '__main__':
    file_names = jpgsIntoList('.')
    run_tests(file_names)
