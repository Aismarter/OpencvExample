import cv2 as cv
import numpy as np


def canny_demo(image):
    t = 80
    canny_output = cv.Canny(image, t, t * 2)  # canny算子
    cv.imshow("canny_output", canny_output)  # 展示
    cv.imwrite("canny_output.png", canny_output)
    return canny_output


src = cv.imread("./pictures/robot.jpg")  # 读
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
binary = canny_demo(src)
k = np.ones((3, 3), dtype=np.uint8)
binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)
# morphologyEx函数利用基本的膨胀和腐蚀技术，来执行更加高级形态学变换

# 轮廓发现

contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for c in range(len(contours)):
    rect = cv.minAreaRect(contours[c])  # 返回矩阵点
    cx, cy = rect[0]
    ww, hh = rect[1]
    ratio = np.minimum(ww, hh) / np.maximum(ww, hh)
    print(ratio)
    mm = cv.moments(contours[c])
    m00 = mm['m00']
    m10 = mm['m10']
    m01 = mm['m01']
    cx = np.int(m10 / m00)
    cy = np.int(m01 / m00)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    if ratio > 0.9:
        cv.drawContours(src, [box], 0, (0, 255, ), 2)
        cv.circle(src, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), 2, 8, 0)
    if ratio < 0.5:
        cv.drawContours(src, [box], 0, (255, 0, 255), 2)
        cv.circle(src, (np.int32(cx), np.int32(cy)), 2, (0, 0, 255), 2, 8, 0)

# 显示
cv.imshow("contours_analysis", src)
cv.imwrite("contours_analysis.png", src)
cv.waitKey(0)
cv.destroyAllWindows()
