import cv2
import numpy as np
import onnx
import time

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
# change this for main confidence
CONFIDENCE_THRESHOLD = 0.001

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors.
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)

# some functions we'll need
CAMERA_BUFFER_SIZE = 3


def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(
        im, (x, y), (x + dim[0], y + dim[1] + baseline), (0, 0, 0), cv2.FILLED)
    # Display text inside the rectangle.
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE,
                FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def format_img_for_yolov5(source):
    MODEL_IMG_SIZE = (INPUT_WIDTH, INPUT_HEIGHT)
    # put the image in square big enough
    # col, row, _ = source.shape
    # _max = max(col, row)
    # resized = np.zeros((_max, _max, 3), np.uint8)
    # resized[0:col, 0:row] = source
    # resize to 640x640, normalize to [0,1[ and swap Red and Blue channels
    result = cv2.dnn.blobFromImage(
        source, 1/255.0, MODEL_IMG_SIZE, swapRB=True)
    return result


def pre_process(input_image, net):
    # Create a 4D blob from a frame.
    net.setInput(format_img_for_yolov5(input_image))
    # Run the forward pass to get output of the output layers.
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs


def save_cropped_image(crop_img, file_name):
    cv2.imshow('Output', crop_img)
    # make frames dic first makedir frames
    # print(file_name)
    cv2.waitKey(2000)
    cv2.imwrite(file_name, crop_img)


def post_process_and_save(input_image, outputs, file_name_no_ex="01", extn=".jpg"):
    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []
    # Rows.
    rows = outputs[0].shape[1]
    # print(f"Rows found: {rows}")
    image_height, image_width = input_image.shape[:2]
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
    if len(indices) > 0:
        max_confidence = max(confidences)
        max_index = confidences.index(max_confidence)
        box = boxes[max_index]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # Draw bounding box.
        output_img = input_image.copy()
        cv2.rectangle(output_img, (left, top),
                      (left + width, top + height), BLUE, 3*THICKNESS)
        # Class label.
        label = "{}:{:.2f}".format(
            classes[class_ids[max_index]], confidences[max_index])
        # Draw label.
        draw_label(output_img, label, left, top)
        cropped_top = int(top-height/32)
        cropped_bottom = int(top+height+height/32)
        cropped_left = int(left-width/32)
        cropped_right = int(left+width+width/32)
        crop_img = input_image[cropped_top:cropped_bottom,
                               cropped_left:cropped_right]
        if crop_img.shape[0] > 5 and crop_img.shape[1] > 20:
            save_cropped_image(crop_img, file_name_no_ex +
                               f"{max_confidence:05f}" + extn)
            return crop_img
    # for idx, i in enumerate(indices):
    #     box = boxes[i]
    #     left = box[0]
    #     top = box[1]
    #     width = box[2]
    #     height = box[3]
    #     cropped_top = int(top-height/32)
    #     cropped_bottom = int(top+height+height/32)
    #     cropped_left = int(left-width/32)
    #     cropped_right = int(left+width+width/32)
    #     crop_img = input_image[cropped_top:cropped_bottom,
    #                            cropped_left:cropped_right]
    #     if crop_img.shape[0] > 5 and crop_img.shape[1] > 20:
    #         save_cropped_image(crop_img, file_name_no_ex + f"{idx:02d}" + extn)
    #         return crop_img
    return input_image


def post_process(input_image, outputs):
    # Lists to hold respective values while unwrapping.
    output_img = input_image.copy()
    class_ids = []
    confidences = []
    boxes = []
    # Rows.
    rows = outputs[0].shape[1]
    print(f"Rows found: {rows}")
    image_height, image_width = input_image.shape[:2]
    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT
    # Iterate through detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        # print("Confidence:", confidence)
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
    # print("Total indices: ", len(indices))
    if len(indices) > 0:
        max_confidence = max(confidences)
        max_index = confidences.index(max_confidence)
        box = boxes[max_index]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # Draw bounding box.
        cv2.rectangle(output_img, (left, top),
                      (left + width, top + height), BLUE, 3*THICKNESS)
        # Class label.
        label = "{}:{:.2f}".format(
            classes[class_ids[max_index]], confidences[max_index])
        # Draw label.
        draw_label(output_img, label, left, top)
        cropped_top = int(top-height/32)
        cropped_bottom = int(top+height+height/32)
        cropped_left = int(left-width/32)
        cropped_right = int(left+width+width/32)
        crop_img = input_image[cropped_top:cropped_bottom,
                               cropped_left:cropped_right]
        if crop_img.shape[0] > 5 and crop_img.shape[1] > 20:
            save_cropped_image(crop_img, f"{max_confidence:05f}.jpg")
    # for idx, i in enumerate(indices):
    #     box = boxes[i]
    #     left = box[0]
    #     top = box[1]
    #     width = box[2]
    #     height = box[3]
    #     # Draw bounding box.
    #     cv2.rectangle(output_img, (left, top),
    #                   (left + width, top + height), BLUE, 3*THICKNESS)
    #     # Class label.
    #     label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
    #     # Draw label.
    #     draw_label(output_img, label, left, top)
    #     cropped_top = int(top-height/32)
    #     cropped_bottom = int(top+height+height/32)
    #     cropped_left = int(left-width/32)
    #     cropped_right = int(left+width+width/32)
    #     save_cropped_image(input_image[cropped_top:cropped_bottom,
    #                                    cropped_left:cropped_right], f"{idx:05d}.jpg")
    return output_img


def detectAndShowImage(file_name, net):
    frame = cv2.imread(file_name)
    start = time.time()
    detections = pre_process(frame, net)
    t, _ = net.getPerfProfile()
    # print(detections)
    img = post_process(frame.copy(), detections)
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    print(label)
    cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE,
                (0, 0, 255), THICKNESS, cv2.LINE_AA)
    # stop timer
    end = time.time()
    # show timing information on YOLO
    print("[INFO] Total function took {:.6f} seconds".format(end - start))
    cv2.imshow('Output', img)
    cv2.waitKey(2000)
    # cv2.waitKey(0)


def downsize_frame(frame_input, max_height=1080):
    if frame_input.shape[0] > 1081:
        height = max_height
        width = int(frame_input.shape[1] * height / frame_input.shape[0])
        dim = (width, height)
        frame_input = cv2.resize(
            frame_input, dim, interpolation=cv2.INTER_AREA)
    return frame_input


def detectAndSaveStream(video_source, net):
    cap = cv2.VideoCapture(video_source)
    video_file_name_no_ex = "frames/" + str((video_source)).split('.')[0]
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    frame_index = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            detections = pre_process(downsize_frame(frame), net)
            post_process_and_save(frame,
                                  detections,
                                  video_file_name_no_ex + f"{frame_index:05d}")
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            frame_index += 1
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Load class names.
    # classesFile = "coco.names"
    classes = ["plate", ]
    net = cv2.dnn.readNet('../models/best_simp.onnx')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # Load image.
    # to get onnx model from pt run from yolov5 venv first convertion command
    # python export.py --weights 'imported/best.pt' --include onnx --data 'imported/data.yaml'
    # net = cv2.dnn.readNet('../models/best.onnx')
    # simplify model if error on import of the original
    # Process image.
    # detectAndShowImage('test.png', net)
    detectAndSaveStream('01.ts', net)
