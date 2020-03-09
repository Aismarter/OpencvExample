import cv2 as cv
import numpy as np

src = cv.imread("./pictures/factory.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

# 高斯模糊去噪声
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imwrite("binary1.png", binary)
cv.imshow("binary1", binary)

# 闭操作
se1 = cv.getStructuringElement(cv.MORPH_RECT, (5, 5), (-1, -1))
# getStructuringElement函数会返回指定形状和尺寸的结构元素。
# 这个函数的第一个参数表示内核的形状，有三种形状可以选择。
# 矩形：MORPH_RECT
# 交叉形：MORPH_CROSS
# 椭圆形：MORPH_ELLIPSE
# 第二和第三个参数分别是内核的尺寸以及锚点的位置。
# 一般在调用erode以及dilate函数之前，先定义一个Mat类型的变量来获得getStructuringElement函数的返回值。
# 对于锚点的位置，有默认值Point（-1,-1），表示锚点位于中心点。
# element形状唯一依赖锚点位置，其他情况下，锚点只是影响了形态学运算结果的偏移。

se2 = cv.getStructuringElement(cv.MORPH_RECT, (5, 25), (-1, -1))
binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, se1)
binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, se2)

cv.imshow("binary", binary)
cv.imwrite("binary2.png", binary)

cv.waitKey(0)
cv.destroyAllWindows()
