import cv2
# print(cv2.__version__)
import numpy as np


def image_hystogram(image):
    hist = cv2.calcHist([image], [1], None, [4], [0, 256])
    print(hist)
    return hist


def good_image(image):
    hist = cv2.calcHist([image], [1], None, [4], [0, 256])
    # print(hist)
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
    cv2.imshow("cropped", image[y:y+h, x:x+w])
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    return image[y:y+h, x:x+w]


def contrast_image(image, brightness=-60, contrast=1.25):
    # Adjust the brightness and contrast
    # Adjusts the brightness by adding 10 to each pixel value
    image2 = cv2.addWeighted(image, contrast, np.zeros(
        image.shape, image.dtype), 0, brightness)
    # show for a second
    cv2.imshow("cropped", image2)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    return image2


# check_if_good_image(cv2.imread("sample_cars/grey.jpg"))
# cropped = crop_image(cv2.imread(
#     "sample_cars/for_crop/23sc4_01.jpeg"), 0.09, 0.43, 0.22, 0.15)
# cv2.imwrite("sample_cars/cropped.jpg", cropped)
# print(good_image(cv2.imread("sample_cars/grey2.jpg")))
# good_image(cv2.imread("sample_cars/05.jpg"))
contrasted = contrast_image(cv2.imread("sample_cars/for_contrast/02.jpg"))
