import cv2 as cv

src = cv.imread("./pictures/factory.jpg")

cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
h, w, ch = src.shape  # shape 返回数组的函数与列数
print("h, w, ch", h, w, ch)
for row in range(h):
    for col in range(w):
        b, g, r = src[row, col]
        b = 255 - b
        g = 255 - g
        r = 255 - r  # 取反以后可以获得不同的像素色彩值
        src[row, col] = [b, g, r]

cv.namedWindow("output", cv.WINDOW_AUTOSIZE)
cv.imshow("output", src)

cv.waitKey(0)
cv.destroyAllWindows()

