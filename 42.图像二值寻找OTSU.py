import cv2 as cv
import numpy as np

#
# THRESH_BINARY = 0
# THRESH_BINARY_INV = 1
# THRESH_TRUNC = 2
# THRESH_TOZERO = 3
# THRESH_TOZERO_INV = 4

src = cv.imread("./pictures/factory.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
h, w = src.shape[:2]

# 自动阈值分割 OTSU
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#
# C++ void adaptiveThreshold(
#                              InputArray src,    输入图像.
#                              OutputArray dst,     输出图像.
#                              double max_value,
#                              int adaptive_method,
#                              int threshold_type,
#                              int block_size,
#                              double param )
# 第一个参数，输入图像，需为8位单通道浮点图像
# 第二个参数，输出图像，需和原图像尺寸类型一致
# 第三个参数，double类型，给像素赋的满足条件的非零值
# 第四个参数，用于指定自适应阈值的算法，CV_ADAPTIVE_THRESH_MEAN_C ，CV_ADAPTIVE_THRESH_GAUSSIAN_C    
# 第五个参数，取阈值类型：必须是CV_THRESH_BINARY或者CV_THRESH_BINARY_INV
# 第六个参数，用来计算阈值的象素邻域大小: 3, 5, 7, ...
# 第七个参数，与方法有关的参数。对方法 CV_ADAPTIVE_THRESH_MEAN_C 和 CV_ADAPTIVE_THRESH_GAUSSIAN_C
# 它是一个从均值或加权均值提取的常数, 有时也可以是小数或负数。
# 对方法 CV_ADAPTIVE_THRESH_MEAN_C，先求出块中的均值，再减掉param。
# 对方法 CV_ADAPTIVE_THRESH_GAUSSIAN_C
# 那么区域中（x，y）周围的像素根据高斯函数按照他们离中心点的距离进行加权计算，再减掉param。

print("ret :", ret)
cv.imshow("binary", binary)

result = np.zeros([h, w * 2, 3], dtype=src.dtype)
result[0:h, 0:w, :] = src
result[0:h, w:2 * w, :] = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
cv.putText(result, "input", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.putText(result, "binary, threshold = " + str(ret), (w + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.imshow("result", result)
cv.imwrite("binary_result.png", result)

cv.waitKey(0)
cv.destroyAllWindows()
