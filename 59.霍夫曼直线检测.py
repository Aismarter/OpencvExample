import cv2 as cv
import numpy as np


def canny_demo(image):
    t = 80
    canny_output = cv.Canny(image, t, t * 2)
    cv.imwrite("canny_output.png", canny_output)
    return canny_output


src = cv.imread("./pictures/factory.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

binary = canny_demo(src)
cv.imshow("binary", binary)

lines = cv.HoughLines(binary, 1, np.pi / 3.14, 120, None, 0, 0)
# 此函数可以找出采用标准霍夫变换的二值图像线条。
# HoughLines函数是标准霍夫线性变换函数，该函数的功能是通过一组参数对的集合来表示检测到的直线。
# HoughLines(image, rho, theta, threshold[, lines[, srn[, stn[, min_theta[, max_theta]]]]]) -> lines
# image参数           表示边缘检测的输出图像，该图像为单通道8位二进制图像。
# rho参数             表示参数极径 r 以像素值为单位的分辨率，这里一般使用1像素。
# theta参数           表示参数极角 \theta 以弧度为单位的分辨率，这里使用1度。
# threshold参数       表示检测一条直线所需最少的曲线交点。
# lines参数           表示储存着检测到的直线的参数对 (r,\theta) 的容器 。
# srn参数、stn参数     默认都为0。如果srn = 0且stn = 0，则使用经典的Hough变换。
# min_theta参数       表示对于标准和多尺度Hough变换，检查线条的最小角度。
# max_theta参数       表示对于标准和多尺度Hough变换，检查线条的最大角度。
#
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        print(a)
        b = np.sin(theta)
        print(b)
        x0 = a * rho
        y0 = b * rho
        print("x0 y0:", x0, y0)
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(src, pt1, pt2, (0, 0, 255), 1, cv.LINE_AA)


# 显示
cv.imshow("hough line demo", src)
cv.imwrite("contours_analysis.png", src)
cv.waitKey(0)
cv.destroyAllWindows()
