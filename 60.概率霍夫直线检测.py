# 概率霍夫直线检测
import cv2 as cv
import numpy as np


def canny_demo(image):
    t = 80
    canny_output = cv.Canny(image, t, t * 2)
    return canny_output


src = cv.imread("./pictures/factory.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
binary = canny_demo(src)
cv.imshow("binary", binary)


linesP = cv.HoughLinesP(binary, 1, np.pi / 180, 235, None, 50, 10)
# cv.HoughLinesP 霍夫变化的概论形式
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(src, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 1, cv.LINE_AA)


# 显示
cv.imshow("hough line demo", src)
cv.waitKey(0)
cv.destroyAllWindows()

