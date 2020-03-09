# 缺陷检测
# 检测刀片上裂缝的缺陷
import cv2 as cv
import numpy as np

src = cv.imread("./pictures/ce_02.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# 参数说明：
# 第一个原图像，第二个进行分类的阈值，第三个是高于（低于）阈值时赋予的新值，
# 第四个是一个方法选择参数，常用的有：
# • cv2.THRESH_BINARY（黑白二值）
# • cv2.THRESH_BINARY_INV（黑白二值反转）
# • cv2.THRESH_TRUNC （得到的图像为多像素值）
# • cv2.THRESH_TOZERO
# • cv2.THRESH_TOZERO_INV
# 该函数有两个返回值，第一个retVal（得到的阈值值（在后面一个方法中会用到）），第二个就是阈值化后的图像。

se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))
# getStructuringElement函数会返回指定形状和尺寸的结构元素。
# 这个函数的第一个参数表示内核的形状，有三种形状可以选择。
# 矩  形： MORPH_RECT;
# 交叉形： MORPH_CROSS;
# 椭圆形： MORPH_ELLIPSE;
# 第二和第三个参数分别是内核的尺寸以及锚点的位置。
binary = cv.morphologyEx(binary, cv.MORPH_GRADIENT, se)  # 对binary图像做开运算
# cv.MORPH_RECT(白底) 无法检测除缺陷 MORPH_OPEN(白底) MORPH_HITMISS(白底) MORPH_ELLIPSE(白底) MORPH_ERODE(白底)
# cv.MORPH_BLACKHAT(黑底，无明显特征) 无法检测出区域 MORPH_TOPHAT(黑底，缺陷区域有特征)
# cv.MORPH_CROSS 可以识别出缺陷 MORPH_DILATE MORPH_GRADIENT(黑底，检测结果为边缘)
cv.imshow("binary", binary)


# 观察图像与提取图像ROI对象轮廓外接矩形与轮廓.
contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
height, width = src.shape[:2]
for c in range(len(contours)):
    x, y, w, h = cv.boundingRect(contours[c])
    # 用一个最小的矩形，把找到的形状包起来。
    # 返回四个值，分别是x，y，w，h；
    # x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
    area = cv.contourArea(contours[c])  # 计算轮廓面积
    if h > (height//2):
        continue
    if area < 150:
        continue
    cv.rectangle(src, (x, y), (x+w, y+h), (0, 0, 255), 1, 8, 0)  # 画出矩形框
    cv.drawContours(src, contours, c, (0, 255, 0), 2, 8)  # 画出点

cv.imshow("result", src)
cv.imwrite("binary2.png", src)

cv.waitKey(0)
cv.destroyAllWindows()
