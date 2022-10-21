import cv2
import math
import numpy as np
from numpy import fft
import cmath

# +-128 for brightness, +-64 for contrast


def apply_brightness_contrast(img, brightness=-10, contrast=10):
    input_img = img
    # input_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if cv2.mean(input_img)[0] < 128:
    #     return img
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf


def calcPSF(roi, len, theta):
    # h = cv2.Mat(filterSize, cv2.CV_32F)
    filterSize = * roi, 1
    # print(filterSize)
    h = np.zeros(filterSize, dtype="float32")
    # center_coordinates
    point = (int(filterSize[1] / 2), int(filterSize[0] / 2))
    ellipse = cv2.ellipse(h, point, (0, round(float(len) / 2.0)),
                          90.0 - theta, 0, 360, float(255), cv2.FILLED)
    # print(ellipse)
    # for BGR use following
    # ellipse = cv2.ellipse(h, point, (0, round(float(len) / 2.0)),
    #                       90.0 - theta, 0, 360, (255,255,255), cv2.FILLED)
    # print(np.sum(ellipse[:, :, 0]))
    outputImg = ellipse / np.sum(ellipse[:, :, 0])
    # print(outputImg)
    # cv2.imshow("outputImg", outputImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return outputImg


def fftshift(inputImg):
    outputImg = inputImg.copy()
    cx = int(outputImg.shape[1] / 2)
    cy = int(outputImg.shape[0] / 2)
    # print(cy, cx, outputImg.shape)
    q0 = outputImg[0:cy, 0:cx].copy()
    # print("q0", q0.shape)
    q1 = outputImg[0:cy, cx:].copy()
    # print("q1", q1.shape)
    q2 = outputImg[cy:, 0:cx]
    # print("q2", q2.shape)
    q3 = outputImg[cy:, cx:]
    # print("q3", q3.shape)
    outputImg[0:cy, 0:cx] = q3
    outputImg[cy:, cx:] = q0
    outputImg[0:cy, cx:] = q2
    outputImg[cy:, 0:cx] = q1
    # cv2.imshow("inputImg", inputImg)
    # cv2.imshow("outputImg", outputImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return outputImg


def calcWnrFilter(h_PSF, nsr):
    h_PSF_shifted = fftshift(h_PSF)
    return fftshift


def deblur_motion_wiener(img, len, theta, snr):
    # print("img shape color", img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print("img shape gray", img.shape)
    rounded_height = math.floor(img.shape[0] / 2) * 2
    rounded_width = math.floor(img.shape[1] / 2) * 2
    # print("rounded_hight, rounded width", rounded_height, rounded_width)
    roi = (rounded_height, rounded_width)
    # rounded_image = img[:rounded_height, :rounded_width]
    # print("rounded_img shape", rounded_image.shape)
    h = calcPSF(roi, len, theta)
    calcWnrFilter(h, 1.0 / snr)
    # print(h)
    return img


def motion_process_psf(image_size, motion_angle):
    PSF = np.zeros(image_size)
    print(image_size)
    center_position = (image_size[0] - 1) / 2
    print(center_position)
    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if slope_tan <= 1:
        for i in range(15):
            offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
            PSF[int(center_position + offset),
                int(center_position - offset)] = 1
        return PSF / PSF.sum()  # Normalize the luminance of the point spread function
    else:
        for i in range(15):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset),
                int(center_position + offset)] = 1
        return PSF / PSF.sum()


def inverse(input, PSF, eps):  # Inverse filtering
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps  # Noise power, that's given，consider epsilon
    # Compute the inverse Fourier transform of F(u,v)
    result = fft.ifft2(input_fft / PSF_fft)
    result = np.abs(fft.fftshift(result))
    return result


def wiener(input, PSF, eps, K=0.01):  # Wiener filtering，K=0.01
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result


def degradation_function(m, n, a, b, T):
    P = m / 2 + 1
    Q = n / 2 + 1
    Mo = np.zeros((m, n), dtype=complex)
    for u in range(m):
        for v in range(n):
            temp = cmath.pi * ((u - P) * a + (v - Q) * b)
            if temp == 0:
                Mo[u, v] = T
            else:
                Mo[u, v] = T * cmath.sin(temp) / temp * cmath.exp(- 1j * temp)
    return Mo


def image_mapping(image):
    img = image/np.max(image)*255
    return img


def deblur_motion(img, a=1, b=1, T=1, r=1):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m, n = img.shape
    # a = a*1e-5
    # b = b*0.1
    # T = T*0.1
    # r = r*0.000005
    a = a*1e-5 / 20
    b = b*0.1 / 20
    T = T*0.1 / 10
    r = r*0.000005 / 40
    G = fft.fft2(img)
    G_shift = fft.fftshift(G)
    H = degradation_function(m, n, a, b, T)
    p = np.array([[0, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 0]])
    P = fft.fft2(p, [img.shape[0], img.shape[1]])
    F = G_shift * (np.conj(H) / (np.abs(H)**2+r*np.abs(P)**2))
    f_pic = np.abs(fft.ifft2(F))
    result = image_mapping(f_pic)
    res1 = result.astype('uint8')
    # cv2.imshow("original", img)
    # cv2.imshow("deblurredInverse", res1)
    # # cv2.imshow("wiener", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # result = deblur_motion(img, len, theta, snr)
    res1 = apply_brightness_contrast(res1)
    return res1


def testSaveImageApplyDeblur(file_name):
    img = cv2.imread(file_name)
    t = 1
    a = 1
    b = 1
    for r in range(1, 20, 2):
        cv2.imwrite(f"frames/a{a:02d}b{b:02d}t{t:02d}r{r:02d}.jpg",
                    deblur_motion(img, a, b, t, r))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # calcPSF((8, 8), 4, 70)
    # deblur_motion_from_file("car.jpg", 128, 0, 700)
    # res1 = deblur_motion(cv2.imread("95534483_l.jpg"))
    # cv2.imshow("deblurredInverse", res1)
    # # cv2.imshow("wiener", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    testSaveImageApplyDeblur("95534483_l.jpg")
