import cv2 as cv
import numpy as np
#  ORB - (Oriented Fast and Rotated BRIEF)算法是基于FAST特征检测与BRIEF特征描述子匹配实现
#  相比BRIEF算法中依靠随机方式获取而值点对，ORB通过FAST方法，
#  FAST方式寻找候选特征点方式是假设灰度图像像素点A周围的像素存在连续大于或者小于A的灰度值

src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)


# 创建orb检测器
orb = cv.ORB_create()
kps = orb.detect(src)

# -1 表示随机颜色
# result = cv.drawKeypoints(src, kps, None, -1, cv.DrawMatchesFlags_DEFAULT)
result = cv.drawKeypoints(src, kps, None, (0, 255, 0), cv.DrawMatchesFlags_DEFAULT)
cv.imshow("result", result)

cv.waitKey(0)
cv.destroyAllWindows()

