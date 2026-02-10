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
cams["20a"] = {
    "ip": "192.168.120.14",
    "channel": "1", }
cams["20b"] = {
    "ip": "192.168.120.14",
    "channel": "2", }
cams["21a"] = {
    "ip": "192.168.120.14",
    "channel": "3", }
cams["21b"] = {
    "ip": "192.168.120.14",
    "channel": "4", }
cams["22a"] = {
    "ip": "192.168.120.14",
    "channel": "5", }
cams["22b"] = {
    "ip": "192.168.120.14",
    "channel": "6", }
cams["23a"] = {
    "ip": "192.168.120.14",
    "channel": "7", }
cams["23b"] = {
    "ip": "192.168.120.14",
    "channel": "8", }

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
