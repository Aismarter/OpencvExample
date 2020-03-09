import cv2 as cv
import numpy as np

src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

# 高斯模糊去噪声
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
dst = cv.GaussianBlur(gray, (9, 9), 2, 2)
binary_1 = cv.adaptiveThreshold(dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY_INV, 45, 15)
# 自适应阈值操作
# def threshold(src, thresh, maxval, type, dst=None)
# thresh：Double类型的，具体的阈值。
# maxval：Double类型的，阈值的最大值
#
# type:
#       THRESH_BINARY 二进制阈值化 -> 大于阈值为1 小于阈值为0
#       THRESH_BINARY_INV 反二进制阈值化 -> 大于阈值为0 小于阈值为1
#       THRESH_TRUNC 截断阈值化 -> 大于阈值为阈值，小于阈值不变
#       THRESH_TOZERO 阈值化为0 -> 大于阈值的不变，小于阈值的全为0
#       THRESH_TOZERO_INV 反阈值化为0 -> 大于阈值为0，小于阈值不变


# 开操作
se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5), (-1, -1))
binary = cv.morphologyEx(binary_1, cv.MORPH_OPEN, se)

cv.imshow("binary", np.hstack((binary_1, gray, binary)))

cv.waitKey(0)
cv.destroyAllWindows()

