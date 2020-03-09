import cv2 as cv
import numpy as np


src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
k = cv.getStructuringElement(cv.MORPH_CROSS, (8, 8))
# getStructuringElement函数会返回指定形状和尺寸的结构元素。
# 这个函数的第一个参数表示内核的形状，有三种形状可以选择。
# 矩形：MORPH_RECT;
# 交叉形：MORPH_CROSS;
# 椭圆形：MORPH_ELLIPSE;
# 第二和第三个参数分别是内核的尺寸以及锚点的位置。
# 一般在调用erode以及dilate函数之前，先定义一个Mat类型的变量来获得getStructuringElement函数的返回值。
# 对于锚点的位置，有默认值Point（-1,-1），表示锚点位于中心点。
# element形状唯一依赖锚点位置，其他情况下，锚点只是影响了形态学运算结果的偏移。


# 开运算去除外部噪点
binary = cv.morphologyEx(binary, cv.MORPH_OPEN, k)
# # morphologyEx函数利用基本的膨胀和腐蚀技术，来执行更加高级形态学变换
cv.imshow("binary", binary)

# 轮廓发现
# 凸包和轮廓近似相似，但不同，虽然有些情况下它们给出的结果是一样的。
# 一般来说，凸性曲线总是凸出来的，至少是平的。如果有地方凹进去了就被叫做凸性缺陷。
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))
for c in range(len(contours)):
    # 凸性检测 检测一个曲线是不是凸的
    ret = cv.isContourConvex(contours[c])
    # 函数cv.convexHull()可以用来检测一个曲线是否具有凸性缺陷，并能纠正缺陷
    points = cv.convexHull(contours[c])
    # cv.polylines(src, [points], True, (0, 255, 0), 2)
    total = len(points)
    print(total)
    for i in range(len(points)):
        x1, y1 = points[i][0]
        # x2, y2 = points[(i + 1) % total][0]
        cv.circle(src, (x1, y1), 6, (255, 0, 0), 1, 8, 0)
        # cv.line(src, (x1, y1), (x2, y2), (0, 0, 255), 1, 8, 0)
    print(points)
    print("convex : ", ret)

# 显示
cv.imshow("contours_analysis", src)
cv.imwrite("contours_analysis.png", src)
cv.waitKey(0)
cv.destroyAllWindows()
