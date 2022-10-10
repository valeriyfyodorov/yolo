import cv2
print(cv2.__version__)
camaddress = {}
camaddress["cam1"] = ["rtsp://admin:AnafigA_123@192.168.20.193:554/media/video1",
                      "rtsp://admin:AnafigA_123@192.168.20.193:554/media/video2"]
camaddress["cam2"] = ["rtsp://admin:AnafigA_123@192.168.20.194:554/media/video1",
                      "rtsp://admin:AnafigA_123@192.168.20.194:554/media/video2"]
print(camaddress["cam1"][0])

cap = cv2.VideoCapture(camaddress["cam1"][0])
if cap.isOpened():
    for i in range(200):
        ret, frame = cap.read()
        if ret and frame.any():
            cv2.imwrite(f"frame{i:03d}.jpg", frame)
        else:
            print(f"skipped {i}")
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
