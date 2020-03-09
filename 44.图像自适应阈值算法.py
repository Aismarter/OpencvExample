import cv2 as cv
import numpy as np

# THRESH_BINARY = 0
# THRESH_BINARY_INV = 1
# THRESH_TRUNC = 2
# THRESH_TOZERO = 3
# THRESH_TOZERO_INV = 4

src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
h, w = src.shape[:2]

# 自动阈值分割 TRIANGLE
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)
# adaptiveThreshold(InputArray src, OutputArray dst, double maxValue,
#                   int adaptiveMethod, int thresholdType, int blockSize, double C)
# InputArray src：源图像
# OutputArray dst：输出图像，与源图像大小一致
# int adaptiveMethod：在一个邻域内计算阈值所采用的算法，有两个取值，
# 分别为 ADAPTIVE_THRESH_MEAN_C 和 ADAPTIVE_THRESH_GAUSSIAN_C 。
# ADAPTIVE_THRESH_MEAN_C的计算方法是计算出领域的平均值再减去第七个参数double C的值
# ADAPTIVE_THRESH_GAUSSIAN_C的计算方法是计算出领域的高斯均值再减去第七个参数double C的值
# int thresholdType：这是阈值类型，只有两个取值，分别为 THRESH_BINARY 和THRESH_BINARY_INV  
# int blockSize：adaptiveThreshold的计算单位是像素的邻域块，邻域块取多大，就由这个值作决定
# double C：在对参数int adaptiveMethod的说明中，我已经说了这个参数的作用，从中可以看出，这个参数实际上是一个偏移值调整量

cv.imshow("binary", binary)

result = np.zeros([h, w * 2, 3], dtype=src.dtype)
result[0:h, 0:w, :] = src
result[0:h, w:2 * w, :] = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
cv.putText(result, "input", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.putText(result, "adaptive threshold", (w + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.imshow("result", result)
cv.imwrite("binary_result.png", result)

cv.waitKey(0)
cv.destroyAllWindows()
