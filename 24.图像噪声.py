# 图像噪声 给图像增加噪点
import cv2 as cv
import numpy as np


def add_salt_pepper_noise(image):
    h, w = image.shape[:2]
    nums = 10000
    rows = np.random.randint(0, h, nums, dtype=np.int)
    cols = np.random.randint(0, w, nums, dtype=np.int)
    # numpy.random.randint(low, high=None, size=None, dtype='l')
    # 函数的作用是，返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)。
    # 如果没有写参数high的值，则返回[0,low)的值。
    # 参数如下：
    # low: int
    # 生成的数值最低要大于等于low。
    # （hign = None时，生成的数值要在[0, low)区间内）
    # high: int (可选)
    # 如果使用这个值，则生成的数值在[low, high)区间。
    # size: int or tuple of ints(可选)
    # 输出随机数的尺寸，比如size = (m * n* k)则输出同规模即m * n* k个随机数。默认是None的，仅仅返回满足要求的单一随机数。
    # dtype: dtype(可选)：
    # 想要输出的格式。如int64、int等等
    for i in range(nums):
        if i % 2 == 1:
            image[rows[i], cols[i]] = (255, 255, 255)
        else:
            image[rows[i], cols[i]] = (0, 0, 0)
    return image


def gaussian_noise(image):
    noise = np.zeros(image.shape, image.dtype)
    m = (15, 15, 15)  # 这里因为是3通道的图像，每个通道都要设置均值与标准差
    s = (30, 30, 30)
    cv.randn(noise, m, s)
    # randn(dst, mean, stddev)
    # dst 随机数输出数组；数组必须预先分配，并有1到4个通道。
    # mean 生成随机数的平均值（期望值）。
    # stddev 生成的随机数的标准差；它可以是向量（在这种情况下，假定为对角标准差矩阵）或平方矩阵。
    dst = cv.add(image, noise)
    return dst


src = cv.imread("./pictures/factory.jpg")
h, w = src.shape[:2]
copy = np.copy(src)
copy1 = add_salt_pepper_noise(copy)
copy2 = gaussian_noise(copy)
result1 = np.zeros([h, w * 2, 3], dtype=src.dtype)
result2 = np.zeros([h, w * 2, 3], dtype=src.dtype)
result1[0:h, 0:w, :] = src
result1[0:h, w:2 * w, :] = copy1
result2[0:h, 0:w, :] = src
result2[0:h, w:2 * w, :] = copy2
cv.putText(result1, "original image", (20, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
cv.putText(result1, "salt pepper image", (w + 20, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
cv.putText(result2, "original image", (20, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
cv.putText(result2, "gaussian noise image", (w + 20, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
cv.imshow("salt pepper noise", result1)
cv.imshow("gaussian noise image", result2)

cv.waitKey(0)
cv.destroyAllWindows()
