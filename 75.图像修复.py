import cv2 as cv
import numpy as np


src = cv.imread("./pictures/xiufu.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)


# 提取划痕
hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv, (100, 43, 46), (124, 255, 255))  # 指定颜色进行掩码操作
cv.imshow("mask", mask)

# 消除
se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
mask = cv.dilate(mask, se)
result = cv.inpaint(src, mask, 3, cv.INPAINT_TELEA)
cv.imshow("result", result)


cv.waitKey(0)
cv.destroyAllWindows()

