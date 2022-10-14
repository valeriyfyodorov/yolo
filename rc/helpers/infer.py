import cv2
import numpy as np
import time

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
# change this for main confidence
CONFIDENCE_THRESHOLD = 0.3


def format_img_for_yolov5(source):
    MODEL_IMG_SIZE = (INPUT_WIDTH, INPUT_HEIGHT)
    result = cv2.dnn.blobFromImage(
        source, 1/255.0, MODEL_IMG_SIZE, swapRB=True)
    return result


def pre_process(input_image, net):
    # Create a 4D blob from a frame.
    net.setInput(format_img_for_yolov5(input_image))
    # Run the forward pass to get output of the output layers.
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs


def post_process(input_image, outputs):
    # Lists to hold respective values while unwrapping.
    output_imgs = []
    class_ids = []
    confidences = []
    boxes = []
    # Rows - each row is one detection
    rows = outputs[0].shape[1]
    # print(f"Rows found: {rows}")
    image_height, image_width = input_image.shape[: 2]
    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT
    # Iterate through detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        # Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            # Get the index of max class score.
            class_id = np.argmax(classes_scores)
            #  Continue if the class score is above threshold.
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)
    # Perform non maximum suppression to eliminate redundant, overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for idx, i in enumerate(indices):
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cropped_top = int(top-height/32)
        cropped_bottom = int(top+height+height/32)
        cropped_left = int(left-width/32)
        cropped_right = int(left+width+width/32)
        output_imgs.append(
            input_image[cropped_top:cropped_bottom, cropped_left:cropped_right])
    return output_imgs


if __name__ == '__main__':
    classes = ["plate", ]
    start = time.time()
    frame = cv2.imread('test.jpg')
    net = cv2.dnn.readNet('../models/best_simp.onnx')
    detections = pre_process(frame, net)
    img = post_process(frame.copy(), detections)
    end = time.time()
    print("[INFO] Total function took {:.6f} seconds".format(end - start))
    cv2.imshow('Output', img)
    cv2.waitKey(2000)
    # cv2.waitKey(0)
