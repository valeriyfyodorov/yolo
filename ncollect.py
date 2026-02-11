
import socket
import cv2
import time
import re
from os import listdir, environ
# import os
# os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

rtsp_template = "rtsp://railcar:AnafigA123_@{ip}:554/unicast/c{channel}/s{stream}/live"
stream = 1
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


def isAlive(address):
    try:
        # connect to the host -- tells us if the host is actually
        # reachable
        sock = socket.create_connection((address, 554), 0.25)
        if sock is not None:
            # print('Clossing socket')
            sock.close
        return True
    except OSError:
        pass
    return False


def good_image(image):
    hist = cv2.calcHist([image], [1], None, [4], [0, 256])
    if hist[1][0] == 0 or hist[2][0] == 0:
        return False
    lows = (hist[0][0] / hist[1][0] * 100)
    highs = (hist[3][0] / hist[2][0] * 100)
    if lows < 0.5 and highs < 0.5:
        return False
    return True


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
            "B", "8").replace("G", "9").replace("L", "1").replace("/", "1")
    result = re.match(r'\b\d{6,10}\b', text)
    if result:
        # print(result, text)
        if len(text) > 8:
            text = text[:8]
        elif len(text) == 7:
            text = text.replace("1", "11", 1)
        return True, text
    return False, text


def initiate_models():
    environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en")  # text detection + text recognition
    return ocr


def ocr_txt_from_img(ocr, img, cam):
    crop = cam["crop"]
    cropped = crop_image(img, crop['x'], crop['y'],
                         crop['width'], crop['height'])
    output = ocr.predict(input=cropped)
    for res in output:
        if 'rec_texts' not in res:
            return None
        # print(res['rec_texts'])
        for item in res['rec_texts']:
            has_num, txt = number_in_text(item)
            if has_num:
                if len(txt) > 5:
                    return txt
    return None


def txt_from_cam(ocr, cam, stream):
    result = None
    url = rtsp_template.format(
        ip=cam["ip"],
        channel=cam["channel"],
        stream=stream,
    )
    # print(cam, url)
    if not isAlive(cam["ip"]):
        # print(f"{cam["ip"]} is not alive")
        return None
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        for i in range(4):
            ret, frame = cap.read()
            if ret and frame.any():
                if not good_image(frame):
                    print(f"skipped {cam["ip"]} {i}")
                    continue
                result = ocr_txt_from_img(ocr, frame, cam)
                # print(result)
                if result:
                    break
            else:
                print(f"skipped {i}")
    cap.release()
    cv2.destroyAllWindows()
    return result


def info_and_confidence_from_api(nr):
    result = {"nr": "", "info": "?", "confidence": 0}
    result["nr"] = nr
    result["confidence"] = 1
    return result


def rway_best_result(ocr, rway_cams_pair, stream):
    result = {"nr": "", "info": "?", "confidence": 0}
    for cam in rway_cams_pair:
        nr = txt_from_cam(ocr, cam, stream)
        if not nr:
            continue
        api_result = info_and_confidence_from_api(nr)
        if api_result["confidence"] == 5:
            return api_result
        if api_result["confidence"] > result["confidence"]:
            result = api_result
    # print("Result", result)
    return result


def loop_all_rways_cams(ocr, cams, stream):
    rways_cams = {
        "20": {"cams": [cams["20sc1"], cams["20sc2"]]},
        "21": {"cams": [cams["21sc1"], cams["21sc2"]]},
        "22": {"cams": [cams["22sc3"], cams["22sc4"]]},
        "23": {"cams": [cams["23sc3"], cams["23sc4"]]},
    }
    result = {}
    for key, value in rways_cams.items():
        # print("key", key, "value", value)
        result[key] = rway_best_result(
            ocr, value["cams"], stream)
    return result


ocr = initiate_models()
# print(txt_from_cam(ocr, cams["21sc1"], 1))
print(loop_all_rways_cams(ocr, cams, stream))
# run_ocr(ocr)
# number_in_text("99439945")
