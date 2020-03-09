import cv2 as cv
import numpy as np

src1 = cv.imread("./pictures/factory.jpg")
src2 = cv.imread("./pictures/robot.jpg")
cv.imshow("factory", src1)
cv.imshow("robot", src2)
h, w, ch = src1.shape
print("h, w, ch : ", h, w, ch)

add_result = np.zeros(src1.shape, src1.dtype)  # 加
cv.add(src2, src1, add_result)  # 这里相加的图像像素大小一定要一样
cv.imshow("add_result", add_result)

sub_result = np.zeros(src1.shape, src1.dtype)  # 减
cv.subtract(src2, src1, sub_result)
cv.imshow("sub_result", sub_result)

mul_result = np.zeros(src1.shape, src1.dtype)  # 乘
cv.multiply(src1, src2, mul_result)
cv.imshow("mul_result", mul_result)

div_result = np.zeros(src1.shape, src1.dtype)
cv.divide(src2, src1, div_result)  # 除
cv.imshow("div_result", div_result)

cv.waitKey(0)
cv.destroyAllWindows()

