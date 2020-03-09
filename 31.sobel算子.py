import cv2 as cv
import numpy as np

src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

h, w = src.shape[:2]
x_grad = cv.Sobel(src, cv.CV_32F, 1, 0)  # 这里采用cv.CV_32F 的方法，考虑到Sobel函数求完导后的数据位置
y_grad = cv.Sobel(src, cv.CV_32F, 0, 1)  # 可能不够因此使用这个长度。
# sobel算子是一种带有方向性的过滤器
# dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])
# 第一个参数是需要处理的图像；
# 第二个参数是图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
# dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。
#
# ksize是Sobel算子的大小，必须为1、3、5、7。
# scale是缩放导数的比例常数，默认情况下没有伸缩系数；
# delta是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
# borderType是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。


x_grad = cv.convertScaleAbs(x_grad)
# convertScaleAbs()函数用来 将转化后的图像转化为原来的uint8形式
y_grad = cv.convertScaleAbs(y_grad)
cv.imshow("x_grad", x_grad)
cv.imshow("y_grad", y_grad)

dst = cv.add(x_grad, y_grad, dtype=cv.CV_16S)
# 使用add函数将转化后的图像组合起来
dst = cv.convertScaleAbs(dst)
sobelPicture = cv.imshow("gradient", dst)
cv.imwrite("./pictures/sobelRobot.jpg", sobelPicture)
result = np.zeros([h, w * 2, 3], dtype=src.dtype)
result[0:h, 0:w, :] = src
result[0:h, w:2 * w, :] = dst
cv.imshow("result", result)

cv.waitKey(0)
cv.destroyAllWindows()
