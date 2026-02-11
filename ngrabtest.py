import cv2
# print(cv2.__version__)

import socket


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
    x = int(image.shape[1] * x_percent)
    y = int(image.shape[0] * y_percent)
    w = int(image.shape[1] * width_percent)
    h = int(image.shape[0] * height_percent)
    # # show for a second
    # cv2.imshow("cropped", image[y:y+h, x:x+w])
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()
    return image[y:y+h, x:x+w]


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

# vcap = cv2.VideoCapture(
#     "rtsp://railcar:AnafigA123_@192.168.120.14:554/unicast/c2/s0/live")
# while (1):
#     ret, frame = vcap.read()
#     cv2.imshow('VIDEO', frame)
#     if cv2.waitKey(500) & 0xFF == ord('q'):
#         break


rtsp_template = "rtsp://railcar:AnafigA123_@{ip}:554/unicast/c{channel}/s{stream}/live"
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
        "x": 0.03,
        "y": 0.58,
        "width": 0.24,
        "height": 0.2,
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
        "x": 0.09,
        "y": 0.39,
        "width": 0.45,
        "height": 0.22,
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

stream = 1


for key, value in cams.items():
    url = rtsp_template.format(
        ip=value["ip"],
        channel=value["channel"],
        stream=stream,
    )
    print(key, url)
    if not isAlive(value["ip"]):
        print(f"{key} is not alive")
        continue
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        for i in range(4):
            ret, frame = cap.read()
            # if i < 3:
            #     print(f"skipped {i}")
            #     continue
            if ret and frame.any():
                if not good_image(frame):
                    print(f"skipped {i}")
                    continue
                cv2.imwrite(f"test_frames/{key}_frame{i:03d}.jpg", frame)
                # cv2.imshow('VIDEO', frame)
            else:
                print(f"skipped {i}")
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    cap.release()
    cv2.destroyAllWindows()
