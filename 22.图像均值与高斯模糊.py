import cv2 as cv
import numpy as np
# ### 几种滤波器的对比
# 均值滤波
# 使用模板内所有像素的平均值代替模板中心像素灰度值
# 易收到噪声的干扰，不能完全消除噪声，只能相对减弱噪声
# 中值滤波
# 计算模板内所有像素中的中值，并用所计算出来的中值体改模板中心像素的灰度值
# 对噪声不是那么敏感，能够较好的消除椒盐噪声，但是容易导致图像的不连续性
# 高斯滤波
# 对图像邻域内像素进行平滑时，邻域内不同位置的像素被赋予不同的权值
# 对图像进行平滑的同时，同时能够更多的保留图像的总体灰度分布特征

src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

# 模糊
dst1 = cv.blur(src, (9, 9))  # 均值滤波
dst2 = cv.GaussianBlur(src, (9, 9), sigmaX=15)  # 高斯滤波
dst3 = cv.GaussianBlur(src, (0, 0), sigmaX=15)

cv.imshow("blur ksize=5", dst1)
cv.imshow("gaussian ksize=9", dst2)
cv.imshow("gaussian ksize=0", dst3)

cv.waitKey(0)
cv.destroyAllWindows()

