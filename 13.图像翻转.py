import cv2 as cv
import numpy as np

src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

# x Filp 倒影
dst1 = cv.flip(src, 0)
# cv::flip()支持图像的翻转（上下翻转、左右翻转，以及对角均可）。
cv.imshow("X-flip", dst1)

# y flip
dst2 = cv.flip(src, 1)
cv.imshow("Y-flip", dst2)

# x y flip 对角
dst3 = cv.flip(src, -1)
cv.imshow("X Y-flip", dst3)

# custom y-flip
h, w, ch = src.shape
dst = np.zeros(src.shape, src.dtype)
for row in range(h):
    for col in range(w):
        b, g, r = src[row, col]
        dst[row, w - col - 1] = [b, g, r]  # 手动进行y翻转
cv.imshow("custom-y-flip", dst)

cv.waitKey(0)
cv.destroyAllWindows()

