from ultralytics import YOLO

model = YOLO("yolov8/best.pt")

# results = model(source=0, show=True, conf=0.25, save=True)  # webcam
# results = model(source="path/to/image.jpg", show=True,
#                 conf=0.25, save=True)  # static image
# results = model(source="screen", show=True, conf=0.25,
#                 save=True)  # screenshot of current screen
# results = model(source="https://ultralytics.com/images/bus.jpg",
#                 show=True, conf=0.25, save=True)  # image or video URL
# results = model(source="path/to/file.csv", show=True,
#                 conf=0.25, save=True)  # CSV file
# results = model(source="path/to/video.mp4", show=True,
#                 conf=0.25, save=True)  # video file
# results = model(source="path/to/dir", show=True, conf=0.25,
#                 save=True)  # all images and videos within directory
# results = model(source="path/to/dir/**/*.jpg", show=True,
#                 conf=0.25, save=True)  # glob expression
# results = model(source="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
#                 show=True, conf=0.25, save=True)  # YouTube video URL

results = model(source="test_frames/cam1_frame002.jpg", show=True,
                conf=0.25, save=True)  # static image

print("Bounding boxes of all detected objects in xyxy format:")
print("Bounding boxes of all detected objects in xywh format:")
print("Confidence values of all detected objects:")
print("Class values of all detected objects:")
for r in results:
    print("xyxy:", r.boxes.xyxy)
    print("xywh:", r.boxes.xywh)
    print("conf:", r.boxes.conf)
    print("cls:", r.boxes.cls)
