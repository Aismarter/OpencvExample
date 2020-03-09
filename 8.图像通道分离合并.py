import cv2 as cv

src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

# 蓝色通道为零
mv = cv.split(src)
mv[0][:, :] = 0
dst1 = cv.merge(mv)
cv.imshow("output-blue0", dst1)

# 绿色通道为零
mv = cv.split(src)
mv[1][:, :] = 0
dst2 = cv.merge(mv)
cv.imshow("output-green0", dst2)

#  红色通道为零
mv = cv.split(src)  # opencv 中split函数和merge函数是一对互逆操作
mv[2][:, :] = 0  # split 可以把一幅画像各个通道分离开
dst3 = cv.merge(mv)  # 经过对各个通过单独操作后可以用merge函数合并。
cv.imshow("output-red0", dst3)

cv.mixChannels(src, dst3, [1, 0])
# mixChannels()函数用于将输入数组的指定通道复制到输出数组的指定通道
# src 输入矩阵的向量 dst输出矩阵的向量
# [2, 0] 决定将2通道的数据拷贝到0通道
cv.imshow("output", dst3)

cv.waitKey(0)
cv.destroyAllWindows()
