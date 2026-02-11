
import cv2
import time
import re
from os import listdir
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def crop_image(image, x_percent, y_percent, width_percent, height_percent):
    # print(image.shape)
    x = int(image.shape[1] * x_percent)
    y = int(image.shape[0] * y_percent)
    w = int(image.shape[1] * width_percent)
    h = int(image.shape[0] * height_percent)
    # show for a second
    # cv2.imshow("cropped", image[y:y+h, x:x+w])
    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()
    return image[y:y+h, x:x+w]


def number_in_text(text):
    # result = re.findall(r'\b\d{3,10}\b', text)
    text = text.upper().replace(" ", "").replace(
        "-", "").replace(".", "").replace(",", "").replace(
            ":", "").replace("[P]", "").replace("[P", "").replace("P", "")
    text = text.replace("O", "0").replace(
        "I", "1").replace("Z", "2").replace("S", "5").replace(
            "B", "8").replace("G", "9").replace("L", "1").replace("/", "1").replace("C", "0")
    result = re.match(r'\b\d{6,10}\b', text)
    if result:
        # print(result, text)
        if len(text) > 8:
            text = text[:8]
        elif len(text) == 7:
            text = text.replace("1", "11", 1)
        return True, text
    return False, text


def run_ocr():
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en")  # text detection + text recognition

    # ocr.PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = True
    # ocr = PaddleOCR(use_doc_orientation_classify=True, use_doc_unwarping=True) # text image preprocessing + text detection + textline orientation classification + text recognition
    # ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False) # text detection + textline orientation classification + text recognition
    # ocr = PaddleOCR(
    #     text_detection_model_name="PP-OCRv5_mobile_det",
    #     text_recognition_model_name="PP-OCRv5_mobile_rec",
    #     use_doc_orientation_classify=False,
    #     use_doc_unwarping=False,
    #     use_textline_orientation=False) # Switch to PP-OCRv5_mobile models
    cams = {}
    cams["20sc1"] = {
        "ip": "192.168.120.14",
        "channel": "1",
        "crop": {
            "x": 0.125,
            "y": 0.614,
            "width":  0.24,
            "height": 0.13,
        },
    }
    cams["20sc2"] = {
        "ip": "192.168.120.14",
        "channel": "2",
        "crop": {
            "x": 0.32,
            "y": 0.46,
            "width": 0.29,
            "height": 0.16,
        },
    }
    cams["21sc1"] = {
        "ip": "192.168.120.14",
        "channel": "3",
        "crop": {
            "x": 0.02,
            "y": 0.6,
            "width": 0.26,
            "height": 0.18,
        }, }
    cams["21sc2"] = {
        "ip": "192.168.120.14",
        "channel": "4",
        "crop": {
            "x": 0.05,
            "y": 0.63,
            "width": 0.28,
            "height": 0.27,
        }, }
    cams["22sc3"] = {
        "ip": "192.168.120.14",
        "channel": "6",
        "crop": {
            "x": 0.19,
            "y": 0.40,
            "width": 0.28,
            "height": 0.35,
        }, }
    cams["22sc4"] = {
        "ip": "192.168.120.14",
        "channel": "6",
        "crop": {
            "x": 0.13,
            "y":  0.57,
            "width": 0.35,
            "height": 0.23,
        }, }
    cams["23sc3"] = {
        "ip": "192.168.120.14",
        "channel": "7",
        "crop": {
            "x": 0.14,
            "y": 0.46,
            "width": 0.37,
            "height": 0.16,
        }, }
    cams["23sc4"] = {
        "ip": "192.168.120.14",
        "channel": "8",
        "crop": {
            "x": 0.09,
            "y": 0.43,
            "width": 0.22,
            "height": 0.15,
        }, }
    start = time.time()
    # dir = "test_frames/"
    dir = "sample_cars/for_crop/"
    files = listdir(dir)
    for file in files:
        # print(file)
        if not file.endswith(".jpg") and not file.endswith(".jpeg"):
            continue
        prefix = file[:5]
        if prefix not in cams:
            print("Unknown prefix", prefix)
            continue
        crop = cams[prefix]["crop"]
        path = dir + file
        print(path)
        cropped = crop_image(cv2.imread(
            path), crop['x'], crop['y'], crop['width'], crop['height'])
        # output = ocr.predict(input=path)
        output = ocr.predict(input=cropped)
        for res in output:
            if 'rec_texts' not in res:
                continue
            print(res['rec_texts'])
            for item in res['rec_texts']:
                has_num, txt = number_in_text(item)
                if has_num:
                    print(txt)
            # res.print()
    elapsed = time.time() - start
    elapsed_per_file = elapsed / len(files)
    print("[INFO] OCR took {:.6f} seconds".format(elapsed))
    print("[INFO] OCR took {:.6f} seconds per file".format(elapsed_per_file))


run_ocr()
# number_in_text("99439945")
