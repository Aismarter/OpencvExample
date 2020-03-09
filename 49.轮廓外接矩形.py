# 轮廓外接矩形
import cv2 as cv
import numpy as np


# canny算子
def canny_demo(image):
    t = 80
    canny_output = cv.Canny(image, t, t * 2)
    cv.imshow("canny_output", canny_output)
    cv.imwrite("canny_output.png", canny_output)
    return canny_output


src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
binary = canny_demo(src)
k = np.ones((3, 3), dtype=np.uint8)
binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)
# morphologyEx函数利用基本的膨胀和腐蚀技术，来执行更加高级形态学变换
# src
# 输入图像，图像位深应该为以下五种之一：CV_8U, CV_16U,CV_16S, CV_32F 或CV_64F。
# dst
# 输出图像，需和源图片保持一样的尺寸和类型。
# op
# 表示形态学运算的类型：
# MORPH_OPEN – 开运算（Opening operation）
# MORPH_CLOSE – 闭运算（Closing operation）
# MORPH_GRADIENT - 形态学梯度（Morphological gradient）
# MORPH_TOPHAT - 顶帽（Top hat）
# MORPH_BLACKHAT - 黑帽（Black hat）
# kernel
# 形态学运算的内核。为NULL，使用参考点位于中心3x3的核。一般使用函数getStructuringElement配合这个参数的使用，
# kernel参数填保存getStructuringElement返回值的Mat类型变量。
# anchor
# 锚的位置，其有默认值（-1，-1），表示锚位于中心。
# iterations
# 迭代使用函数的次数，默认值为1。
# borderType
# 用于推断图像外部像素的某种边界模式。注意它有默认值BORDER_CONSTANT。
# borderValue
# 当边界为常数时的边界值，有默认值morphologyDefaultBorderValue()，


# 轮廓发现 用来检测物体的轮廓
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for c in range(len(contours)):
    # x, y, w, h = cv.boundingRect(contours[c]);
    # cv.drawContours(src, contours, c, (0, 0, 255), 2, 8)
    # cv.rectangle(src, (x, y), (x+w, y+h), (0, 0, 255), 1, 8, 0);
    rect = cv.minAreaRect(contours[c])  # 给定2D点集， 寻找最小面积的包围矩形
    cx, cy = rect[0]
    box = cv.boxPoints(rect)  # 寻找盒子的顶点 包含顶点数据
    box = np.int0(box)
    cv.drawContours(src, [box], 0, (0, 0, 255), 1)  # 画点
    cv.circle(src, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), 1, 8, 0)  # 外接圆形

# 显示
cv.imshow("contours_analysis", src)
cv.imwrite("contours_analysis.png", src)
cv.waitKey(0)
cv.destroyAllWindows()
