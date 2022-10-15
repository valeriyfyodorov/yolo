# import keras_ocr
# pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
# pip install "paddleocr>=2.0.1"
# if errors in opencv loop
# pip uninstall opencv-python
# pip uninstall opencv-contrib-python
# pip uninstall opencv-contrib-python-headless
# pip3 install opencv-contrib-python==4.5.5.62
from paddleocr import PaddleOCR

import pytesseract
import cv2
import easyocr
import time
import os
import numpy as np


# def extract_keras(pipeline, image, file):
#     # img = keras_ocr.tools.read(image)
#     # pipeline = keras_ocr.pipeline.Pipeline()
#     prediction_groups = pipeline.recognize([image, ])
#     for prediction in prediction_groups:
#         output = ""
#         for text, box in prediction:
#             output += text
#         res = processText(output)
#         print(res, res == file[:8])

def processText(txt):
    false_flags = [",", '"', " ", "!", "\n", "'"]
    for flag in false_flags:
        txt = txt.replace(flag, "")
    if len(txt) > 8:
        txt = txt[:8]
    return txt


def extract_paddle(ocr, image, file):
    # img_path = './imgs_en/img_12.jpg'
    result = ocr.ocr(image, cls=True)
    res = ""
    for box in result:
        if len(box) > 0:
            if len(box[0]) > 1:
                txt, conf = box[0][1]
                # print(txt, conf)
                res = processText(txt)
    print(res, res == file[:8], file)


def extract_tesser(image, file):
    text = pytesseract.image_to_string(image)
    res = processText(text)
    print(res, res == file[:8], file)


def extract_easy(reader, image, file):
    box = reader.readtext(image)
    res = ""
    if len(box) > 0:
        res = processText(box[0][1])
    print(res, res == file[:8], file)


def auto_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result


def processImage(image, resize=False):
    # # first try to gray
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # (thresh, image) = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)
    # # try resize
    # auto contrast
    # image = auto_contrast(image)
    if image.shape[1] < 80 or image.shape[1] > 300 or resize:
        new_width = 256
        new_height = int(new_width / image.shape[1] * image.shape[0])
        image = cv2.resize(image,
                           (new_width, new_height),
                           interpolation=cv2.INTER_CUBIC)
    # # try dilate / erode
    # kernel = np.ones((3, 3), np.uint8)
    # image = cv2.erode(image, kernel, iterations=1)
    # image = cv2.dilate(image, kernel, iterations=1)
    # cv2.imshow('Black white image', image)
    # cv2.waitKey(2000)
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
    ocr = PaddleOCR(use_angle_cls=True, lang='en', debug=False, show_log=False)
    start = time.time()
    print("Looping paddle")
    start = time.time()
    for file in file_names:
        image = cv2.imread(file)
        extract_paddle(ocr, processImage(image), file)
    print("[INFO] Paddle took {:.6f} seconds".format(time.time() - start))
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


if __name__ == '__main__':
    file_names = jpgsIntoList('.')
    run_tests(file_names)
