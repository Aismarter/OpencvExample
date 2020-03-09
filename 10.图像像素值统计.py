# 图像像素值统计
import cv2 as cv
import numpy as np

src = cv.imread("./pictures/factory.jpg", cv.IMREAD_GRAYSCALE)
# cv.IMREAD_GRAYSCALE始终将图像转换为单通道的灰度图像

cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

min1, max1, minLoc, maxLoc = cv.minMaxLoc(src)
# minMaxLoc寻找矩阵(一维数组当作向量,用Mat定义) 中最小值和最大值的位置.
# 多通道在使用minMaxLoc()函数是不能给出其最大最小值坐标的，因为每个像素点其实有多个坐标，所以是不会给出的
print("min: %.2f, max: %.2f" % (min1, max1))
print("min loc: ", minLoc)
print("max loc: ", maxLoc)

means, stddev = cv.meanStdDev(src)
# cv.meanStdDev 计算矩阵的均值和标准偏差
# src：输入矩阵，这个矩阵应该是1-4通道的，这可以将计算结果存在Scalar_ ‘s中
# mean：输出参数，计算均值
# stddev：输出参数，计算标准差
print("mean: %.2f, stddev: %.2f" % (means, stddev))
src[np.where(src < means)] = 0
# where 函数，对于不同的数组
# 当数组是一维数组时，返回的值是一维的索引，所以只有一组索引数组
# 当数组是二维数组时，满足条件的数组值返回的是值的位置索引，因此会有两组索引数组来表示值的位置
src[np.where(src > means)] = 255  # 对图像小于均值的数值的坐标位置处的数据进行二值化
cv.imshow("binary", src)

cv.waitKey(0)
cv.destroyAllWindows()
