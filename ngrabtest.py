import cv2
# print(cv2.__version__)

import socket


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
cams["cam1"] = {
    "ip": "192.168.120.14",
    "channel": "1", }
cams["cam2"] = {
    "ip": "192.168.120.14",
    "channel": "2", }
cams["cam3"] = {
    "ip": "192.168.120.14",
    "channel": "3", }
cams["cam4"] = {
    "ip": "192.168.120.14",
    "channel": "4", }
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
        for i in range(10):
            ret, frame = cap.read()
            # if i < 3:
            #     print(f"skipped {i}")
            #     continue
            if ret and frame.any():
                cv2.imwrite(f"test_frames/{key}_frame{i:03d}.jpg", frame)
                # cv2.imshow('VIDEO', frame)
            else:
                print(f"skipped {i}")
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    cap.release()
    cv2.destroyAllWindows()
