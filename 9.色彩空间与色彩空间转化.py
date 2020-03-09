# 色彩空间与色彩空间的转化
# （该点的知识值得注意）

import cv2 as cv

src = cv.imread("./pictures/factory.jpg")
cv.namedWindow("rgb", cv.WINDOW_AUTOSIZE)
cv.imshow("rgb", src)

# RGB to HSV
hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
# cvCvtColor是Opencv里的颜色空间转换函数
# 可以实现RGB颜色向HSV等颜色空间转换
# 根据颜色的直观特性由A. R. Smith在1978年创建的一种颜色空间, 也称六角锥体模型(Hexcone Model)。
# 这个模型中颜色的参数分别是：色调（H），饱和度（S），明度（V）。
cv.imshow("hsv", hsv)

# RBG to YUV
yuv = cv.cvtColor(src, cv.COLOR_BGR2YUV)
# YUV是编译true-color颜色空间（color space）的种类。
# Y'UV, YUV, YCbCr，YPbPr等专有名词都可以称为YUV，彼此有重叠。
# “Y”表示明亮度（Luminance或Luma），也就是灰阶值，“U”和“V”表示的则是色度（Chrominance或Chroma），
# 作用是描述影像色彩及饱和度，用于指定像素的颜色
cv.imshow("YUV", yuv)

# RBG to ycrcb
ycrcb = cv.cvtColor(src, cv.COLOR_BGR2YCrCb)
cv.imshow("ycrcb", ycrcb)

src2 = cv.imread("./pictures/robot.jpg")
cv.imshow("src2", src2)
hsv = cv.cvtColor(src2, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv, (0, 0, 0), (255, 255, 135))
# inRange函数， 可以针对多通道，实现二值化功能。
# 各个通道二值化的阈值必须在设定的两个数组值之间。
cv.imshow("mask", mask)

cv.waitKey(0)
cv.destroyAllWindows()


