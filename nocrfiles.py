

import time
import re
from os import listdir
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def has_numbers_in_text(text):
    # result = re.findall(r'\b\d{3,10}\b', text)
    text = text.upper().replace(" ", "").replace(
        "-", "").replace(".", "").replace(",", "").replace(
            ":", "").replace("[", "").replace("]", "")
    text = text.replace("O", "0").replace(
        "I", "1").replace("Z", "2").replace("S", "5").replace("B", "8")
    result = re.match(r'\b\d{4,10}\b', text)
    if result:
        # print(result, text)
        return True, text
    return False, text


def run_ocr():
    from paddleocr import PaddleOCR
    dir = "test_frames/"
    files = listdir(dir)

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
    start = time.time()
    for file in files:
        if not file.endswith(".jpg"):
            continue
        path = dir + file
        print(path)
        output = ocr.predict(input=path)
        for res in output:
            if 'rec_texts' not in res:
                continue
            # print(res['rec_texts'])
            for item in res['rec_texts']:
                has_num, txt = has_numbers_in_text(item)
                if has_num:
                    print(txt)
            # res.print()
    elapsed = time.time() - start
    elapsed_per_file = elapsed / len(files)
    print("[INFO] OCR took {:.6f} seconds".format(elapsed))
    print("[INFO] OCR took {:.6f} seconds per file".format(elapsed_per_file))


run_ocr()
# has_numbers_in_text("99439945")
