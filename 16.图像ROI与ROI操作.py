# 图像ROI与ROI操作

import cv2 as cv
import numpy as np

src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
h, w = src.shape[:2]
print("h w", h, w)

# 获取ROI
cy = h//2  # // 在python中表示整除法·
cx = w//2
print("cy cx", cy, cx)
roi = src[cy+20:cy+150, cx+10:cx+150, :]
cv.imshow("roi", roi)

# copy ROI
image = np.copy(roi)

# modify ROI
roi[:, :, 0] = 0
cv.imshow("result-roi", roi)
cv.imshow("result-src", src)

# modify copy roi
image[:, :, 2] = 0
cv.imshow("copy ROI", image)

# example with ROI - generate mask
src2 = cv.imread("./pictures/robot.jpg")
cv.imshow("src2", src2)
hsv = cv.cvtColor(src2, cv.COLOR_BGR2HSV)
cv.imshow("HSV", hsv)
mask = cv.inRange(hsv, (35, 43, 46), (99, 255, 255))
cv.imshow("Mask", mask)

# extract person ROI
mask = cv.bitwise_not(mask)  # 对图像做逻辑非操作
cv.imshow("bitwise_not", mask)
person = cv.bitwise_and(src2, src2, mask=mask)
cv.imshow("person", person)

# generate background
result = np.zeros(src2.shape, src2.dtype)
result[:, :, 0] = 255
cv.imshow("background-result", result)

# combine background + person
mask = cv.bitwise_not(mask)
dst = cv.bitwise_or(person, result, mask=mask)
dst = cv.add(dst, person)

cv.imshow("dst", dst)

cv.waitKey(0)
cv.destroyAllWindows()




