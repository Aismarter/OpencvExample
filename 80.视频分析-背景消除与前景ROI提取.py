# 视频分析-背景消除与 前景ROI提取
# 通过视频中的背景进行建模，实现背景消除，生成mask图像，通过对mask二值图像分析实现前景活动对象ROI区域
# 的提取， 是很多视频监控分析软件常用的手段之一，该方法很实时！
# 整体步骤如下：
# 1.初始化背景建模对象GMM
# 2.读取视频一帧
# 3.使用背景建模消除生成mask
# 4.对mask进行轮廓分析提取ROI
# 5.绘制ROI对象

import numpy as np
import cv2 as cv


cap = cv.VideoCapture(0)
fgbg = cv.createBackgroundSubtractorMOG2(
    history=500, varThreshold=100, detectShadows=False)


def process(image, opt=1):
    mask = fgbg.apply(image)
    line = cv.getStructuringElement(cv.MORPH_RECT, (1, 5), (-1, -1))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, line)
    cv.imshow("mask", mask)
    # 轮廓最大，发现最大轮廓
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        area = cv.contourArea(contours[c])
        if area < 150:
            continue
        rect = cv.minAreaRect(contours[c])
        cv.ellipse(image, rect, (0, 255, 0), 2, 8)
        cv.circle(image, (np.int32(rect[0][0]), np.int32(rect[0][1])), 2, (255, 0, 0), 2, 8, 0)
    return image, mask


while True:
    ret, frame = cap.read()
    cv.imwrite("input.png", frame)
    cv.imshow("input", frame)
    result, m_ = process(frame)
    cv.imshow("result", result)
    k = cv.waitKey(50) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
