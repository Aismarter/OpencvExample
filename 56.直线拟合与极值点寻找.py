# 直线拟合
import cv2 as cv
import numpy as np


def canny_demo(image):
    t = 80
    canny_output = cv.Canny(image, t, t * 2)
    cv.imwrite("canny_output.png", canny_output)
    return canny_output


src = cv.imread("./pictures/cub1.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

binary = canny_demo(src)
k = np.ones((3, 3), dtype=np.uint8)
binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)
cv.imshow("binary", binary)

# 轮廓发现
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 直线拟合与极值点寻找
for c in range(len(contours)):
    x, y, w, h = cv.boundingRect(contours[c])
    # boundingRect函数是用来计算轮廓的最小外接矩形，通常与findContours函数组合使用，
    # findContours函数用来查找图像的轮廓，boundingRect获取轮廓的最小外接矩形！
    m = max(w, h)
    # if m < 30:
    #     continue
    vx, vy, x0, y0 = cv.fitLine(contours[c], cv.DIST_L1, 0, 0.01, 0.01)
    # 直线拟合
    # output = cv2.fitLine(InputArray  points, distType, param, reps, aeps)
    # InputArray Points: 待拟合的直线的集合，必须是矩阵形式；
    # distType: 距离类型。fitline为距离最小化函数，拟合直线时，要使输入点到拟合直线的距离和最小化。
    #           这里的距离的类型有以下几种：
    #           cv2.DIST_USER : User defined distance
    #           cv2.DIST_L1: distance = |x1-x2| + |y1-y2|
    #           cv2.DIST_L2: 欧式距离，此时与最小二乘法相同
    #           cv2.DIST_C:distance = max(|x1-x2|,|y1-y2|)
    #           cv2.DIST_L12:L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))
    #           cv2.DIST_FAIR:distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998
    #           cv2.DIST_WELSCH: distance = c2/2(1-exp(-(x/c)2)), c = 2.9846
    #           cv2.DIST_HUBER:distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345
    # param： 距离参数，跟所选的距离类型有关，值可以设置为0。
    # reps, aeps： 第5/6个参数用于表示拟合直线所需要的径向和角度精度，通常情况下两个值均被设定为1e-2.

    k = vy/vx
    b = y0 - k*x0
    maxx = 0
    maxy = 0
    miny = 100000
    minx = 0
    for pt in contours[c]:
        px, py = pt[0]
        if maxy < py:
            maxy = py
        if miny > py:
            miny = py
    maxx = (maxy - b) / k
    minx = (miny - b) / k
    cv.line(src, (np.int32(maxx), np.int32(maxy)),
            (np.int32(minx), np.int32(miny)), (0, 0, 255), 2, 8, 0)


# 显示
cv.imshow("contours_analysis", src)
cv.imwrite("contours_analysis.png", src)

cv.waitKey(0)
cv.destroyAllWindows()

