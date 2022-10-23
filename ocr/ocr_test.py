# import keras_ocr
# pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
# pip install "paddleocr>=2.0.1"
# if errors in opencv loop
# pip uninstall opencv-python
# pip uninstall opencv-contrib-python
# pip uninstall opencv-contrib-python-headless
# pip3 install opencv-contrib-python==4.5.5.62
# for no avx machines
# python3 -m pip install paddlepaddle-gpu==2.3.2 -f https://paddlepaddle.org.cn/whl/stable/noavx.html
from paddleocr import PaddleOCR
import pytesseract
import cv2
import easyocr
import math
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
        # print(box)
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


# +-128 for brightness, +-64 for contrast
def apply_brightness_contrast(img, brightness=-80, contrast=80):
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.mean(input_img)[0] < 128:
        return img
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return apply_filter(buf)


# sharpening filter
def apply_filter(image):
    kernel2d = np.array([[0, 0,  0],
                         [-1,  4, -1],
                         [0, 0,  0]])
    result = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2d)
    return result


def testDisplaySharpenedImage(file_name):
    contrasted = apply_brightness_contrast(cv2.imread(file_name))
    # filtered = apply_filter(contrasted)
    filtered = apply_filter(contrasted)
    cv2.imshow('Contrasted', contrasted)
    cv2.imshow('Filtered', filtered)
    cv2.waitKey()
    cv2.destroyAllWindows()


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
    image = apply_brightness_contrast(image)
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
    ocr = PaddleOCR(use_angle_cls=True, lang='en', debug=True, show_log=True)
    # ocr = PaddleOCR(use_angle_cls=True, lang='en', debug=False, show_log=False)
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


def displayApplyBrigtnessContrast(file_name, brightness, contrast):
    img = cv2.imread(file_name)
    cv2.imshow(file_name, apply_brightness_contrast(img, brightness, contrast))
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def testSaveImageApplyBrigtnessContrast(file_name):
    img = cv2.imread(file_name)
    for b in range(-80, -60, 4):
        for c in range(80, 120, 4):
            cv2.imwrite(f"frames/b{b:03d}c{c:03d}.jpg",
                        apply_brightness_contrast(img, b, c))
    cv2.destroyAllWindows()


def run_save_correction_tests(file_names, brightness, contrast):
    for file in file_names:
        testSaveImageApplyBrigtnessContrast(file, brightness, contrast)


if __name__ == '__main__':
    file_names = jpgsIntoList('.')
    # displayApplyBrigtnessContrast("95534483_l.jpg", -85, 85)
    # run_save_correction_tests(file_names, -64, 96)
    # testSaveImageApplyBrigtnessContrast("95534483_l.jpg")
    # testDisplaySharpenedImage("95534483_l.jpg")
    # calcPSF((8, 8), 4, 70)
    # deblur_motion_from_file("car.jpg", 128, 0, 700)
    run_tests(file_names)
