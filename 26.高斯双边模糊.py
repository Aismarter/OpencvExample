#  bitlateral 图像双边滤波
#  图像去噪的方法很多，如中值滤波，高斯滤波，维纳滤波等等。
#  但这些降噪方法容易模糊图片的边缘细节，对于高频细节的保护效果并不明显。
#  相比较而言，bilateral filter双边滤波器可以很好的边缘保护，即可以在去噪的同时，保护图像的边缘特性。
#  双边滤波（Bilateral filter）是一种非线性的滤波方法，是结合图像的空间邻近度和像素值相似度的一种折衷处理，
#  同时考虑空域信息和灰度相似性，达到保边去噪的目的（不理解这几个概念没关系，后面会慢慢解释）。
import cv2 as cv
import numpy as np

src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

h, w = src.shape[:2]
dst = cv.bilateralFilter(src, 0, 100, 20)
# 函数原型
# void cv::bilateralFilter(InputArray src,
# OutputArray 	dst,
# int 	d,
# double 	sigmaColor,
# double 	sigmaSpace,
# int 	borderType = BORDER_DEFAULT
# )
# InputArray src: 输入图像，可以是Mat类型，图像必须是8位或浮点型单通道、三通道的图像。 
# OutputArray dst: 输出图像，和原图像有相同的尺寸和类型。 
# int d: 表示在过滤过程中每个像素邻域的直径范围。如果这个值是非正数，则函数会从第五个参数sigmaSpace计算该值。 
# double sigmaColor: 颜色空间过滤器的sigma值，这个参数的值月大，表明该像素邻域内有越宽广的颜色会被混合到一起，产生较大的半相等颜色区域。 （这个参数可以理解为值域核的）
# double sigmaSpace:
# 坐标空间中滤波器的sigma值，如果该值较大，则意味着越远的像素将相互影响，
# 从而使更大的区域中足够相似的颜色获取相同的颜色。
# 当d>0时，d指定了邻域大小且与sigmaSpace无关，否则d正比于sigmaSpace. （这个参数可以理解为空间域核的）
# int borderType=BORDER_DEFAULT: 用于推断图像外部像素的某种边界模式，有默认值BORDER_DEFAULT.

result = np.zeros([h, w*2, 3], dtype=src.dtype)
result[0:h, 0:w, :] = src
result[0:h, w:2*w, :] = dst
result = cv.resize(result, (w*2, h))
cv.imshow("result", result)

cv.waitKey(0)
cv.destroyAllWindows()


cv.waitKey(0)
cv.destroyAllWindows()

