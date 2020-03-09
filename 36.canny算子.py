import cv2 as cv
import numpy as np

src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

# t1 = 100, t2 = 3*t1 = 300
edge = cv.Canny(src, 40, 60)
# void cvCanny( const CvArr* image,CvArr*edges,double threshold1,double threshold2, int aperture_size=3 )
# image 输入单通道图像（可以是彩色图像）对于多通道的图像可以用cvCvtColor(将图像从一个颜色空间转换到另一个颜色空间的转换)修改。
# edges 输出的边缘图像，也是单通道的，但是是黑白的
# threshold1 第一个阈值（低阈值）
# threshold2 第二个阈值（高阈值）
# aperture_size Sobel 算子内核大小（滤波计算矩阵的大小默认为3）可以是1、3、5、7
cv.imshow("mask image", edge)  # 展示提取出的边缘
cv.imwrite("./edge.png", edge)
edge_src = cv.bitwise_and(src, src, mask=edge)  # 将边缘于原图做逻辑运算

h, w = src.shape[:2]
result = np.zeros([h, w * 2, 3], dtype=src.dtype)
result[0:h, 0:w, :] = src
result[0:h, w:2 * w, :] = edge_src
cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.putText(result, "edge image", (w + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.imshow("edge detector", result)
cv.imwrite("./result.png", result)

cv.waitKey(0)
cv.destroyAllWindows()
