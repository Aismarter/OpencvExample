import cv2 as cv
import numpy as np


def gaussian_noise(image):
    noise = np.zeros(image.shape, image.dtype)
    m = (15, 15, 15)
    s = (30, 30, 30)
    cv.randn(noise, m, s)  # 生成noise ， m, s 维的数据， 产生高斯白噪声
    # randn(m,n)
    # 生成m×n随机矩阵，其元素服从均值为0，方差为1的标准正态分布。
    # 因此randn函数常用来产生高斯白噪声信号
    # Y=randn(size(A))
    # 生成一个与A维数相同的随机数组，其元素服从均值为0，方差为1的标准正态分布。
    # Y=randn('state')获取一个正态分布产生器当前状态的二元向量。
    # randn('state',0)重新设置正态分布产生器为它的原始状态。
    # randn('state',j)重新设置正态分布产生器为它的第j个状态。

    dst = cv.add(image, noise)  # 加入高斯噪点
    cv.imshow("gaussian noise", dst)
    return dst


src = cv.imread("./pictures/factory.jpg")
cv.imshow("input", src)
h, w = src.shape[:2]
src = gaussian_noise(src)
result3 = cv.medianBlur(src, 5)  # 中值滤波
# ### medianBlur(InputArray src, OutputArray dst, int ksize)
# InputArray src:
# 输入图像，图像为1、3、4通道的图像，当模板尺寸为3或5时，图像深度只能为CV_8U、CV_16U、CV_32F中的一个，如而对于较大孔径尺寸的图片，图像深度只能是CV_8U。
# OutputArray dst:
# 输出图像，尺寸和类型与输入图像一致，可以使用Mat::Clone以原图像为模板来初始化输出图像dst
# . int ksize: 滤波模板的尺寸大小，必须是大于1的奇数，如3、5、7……

cv.imshow("result-medianBlur", result3)

result4 = cv.fastNlMeansDenoisingColored(src, None, 15, 15, 10, 30)  # 图像去噪
# 1. cv2.fastNlMeansDenoising() 使用对象为灰度图。
# 2. cv2.fastNlMeansDenoisingColored() 使用对象为彩色图。
# 3. cv2.fastNlMeansDenoisingMulti() 适用于短时间的图像序列(灰度图像) 
# 要对一段视频使用这个方法。
# 第一个参数是一个噪声帧的列表。
# 第二个参数 imgtoDenoiseIndex 设定那些帧需要去噪,我们可以传入一个帧的索引。
# 第三个参数 temporaWindowSize 可以设置用于去噪的相邻帧的数目,它应该是一个奇数。
# 在这种情况下 temporaWindowSize 帧的图像会被用于去噪,中间的帧就是要去噪的帧
# 4. cv2.fastNlMeansDenoisingColoredMulti() 适用于短时间的图像序列(彩色图像) 
# 共同参数有: 
# • h : 决定过滤器强度。h 值高可以很好的去除噪声,但也会把图像的细节抹去。(取 10 的效果不错) 
# • hForColorComponents : 与 h 相同,但使用与彩色图像。(与 h 相同) 
# • templateWindowSize : 奇数。(推荐值为 7) 
# • searchWindowSize : 奇数。(推荐值为 21)

cv.imshow("result-fastN1MeansDenoisingColored", result4)

cv.waitKey(0)
cv.destroyAllWindows()
