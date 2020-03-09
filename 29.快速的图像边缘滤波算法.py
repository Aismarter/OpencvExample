# 快速的图像边缘滤波算法（快速的边缘保留滤波算法）
# 保边滤波器(Edge Preserving Filter)是指在滤波过程中能够有效的保留图像中的边缘信息的一类特殊滤波器。
# 其中双边滤波器（Bilateral filter）、引导滤波器（Guided image filter）、
# 加权最小二乘法滤波器（Weighted least square filter）为几种比较广为人知的保边滤波器。
# 高斯双边模糊与mean shift均值模糊两种边缘保留滤波算法，都因为计算量比较大，无法实时实现图像边缘保留滤波，
# 限制了它们的使用场景，OpenCV中还实现了一种快速的边缘保留滤波算法。
# 高斯双边与mean shift均值在计算时候使用五维向量是其计算量大速度慢的根本原因，
# 该算法通过等价变换到低纬维度空间，实现了数据降维与快速计算

import cv2 as cv
import numpy as np

src = cv.imread("./pictures/robot.jpg")
# cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
# # cv.imshow("input", src)

h, w = src.shape[:2]
dst = cv.edgePreservingFilter(src, sigma_s=100, sigma_r=0.4, flags=cv.RECURS_FILTER)
# 函数原型
# void edgePreservingFilter( InputArray src, OutputArray dst, int flags = 1,
#                            float sigma_s = 60, float sigma_r = 0.4f);
# src： 输入 8 位 3 通道图像。
# dst： 输出 8 位 3 通道图像。
# flags： 边缘保护滤波  cv::RECURS_FILTER 或 cv::NORMCONV_FILTER。
# sigma_s：取值范围为 0～200。
# sigma_r：取值范围为 0～1。
# 当sigma_s 取值不变时候，sigma_r 越大图像滤波效果越明显；
# 当sigma_r 取值不变时候，窗口 sigma_s 越大图像模糊效果越明显；
# 当sgma_r取值很小的时候，窗口 sigma_s 取值无论如何变化，图像双边滤波效果都不好！
result = np.zeros([h, w * 2, 3], dtype=src.dtype)
result[0:h, 0:w, :] = src
result[0:h, w:w * 2, :] = dst
result = cv.resize(result, (w * 2, h))
cv.imshow("result", result)

cv.waitKey(0)
cv.destroyAllWindows()
