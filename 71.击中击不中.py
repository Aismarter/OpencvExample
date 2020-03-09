import cv2 as cv
import numpy as np

# Hit-miss 算法步骤：
# 击中击不中变换是形态学中用来检测特定形状所处位置的一个基本工具。
# 它的原理就是使用腐蚀；如果要在一幅图像A上找到B形状的目标，我们要做的是：
# 首先，建立一个比B大的模板W；使用此模板对图像A进行腐蚀，得到图像假设为Process1;
# 其次，用B减去W，从而得到V模板(W-B)；使用V模板对图像A的补集进行腐蚀，得到图像假设为Process2;
# 然后，Process1与Process2取交集；得到的结果就是B的位置。
# 这里的位置可能不是B的中心位置，要视W-B时对齐的位置而异；
# 其实很简单，两次腐蚀，然后交集，结果就出来了。


src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

# Binary image
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

# Hit and Miss
se = cv.getStructuringElement(cv.MORPH_CROSS, (12, 12))
binary = cv.morphologyEx(binary, cv.MORPH_HITMISS, se)


cv.imshow("hit miss", binary)
cv.imwrite("binary2.png", binary)

cv.waitKey(0)
cv.destroyAllWindows()

