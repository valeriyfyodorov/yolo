import keras_ocr
import pytesseract
import cv2
import easyocr


def extract_keras(pipeline, file_names):
    images = [
        keras_ocr.tools.read(file_name) for file_name in file_names
    ]
    prediction_groups = pipeline.recognize(images)
    for prediction in prediction_groups:
        output = ""
        for text, box in prediction:
            output += text
        print(output)


def extract_tesser(file_names):
    for img in file_names:
        text = pytesseract.image_to_string(img)
        print(text)


def extract_easy(file_names):
    reader = easyocr.Reader(['en'])
    for img in file_names:
        box = reader.readtext(img)
        print(box[0][1])


if __name__ == '__main__':
    file_names = [
        "00000.jpg",
        "00001.jpg",
        "00002.jpg",
        "00003.jpg",
        "00004.jpg",
    ]
    pipeline = keras_ocr.pipeline.Pipeline()
    print("Starting keras")
    # extract_keras(pipeline, file_names)
    print("Starting tesser")
    # extract_tesser(file_names)
    print("Starting easy")
    extract_easy(file_names)
