import cv2 as cv
import numpy as np


src = cv.imread("./pictures/cub1.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imshow("binary", binary)

# 轮廓发现
image = np.zeros(src.shape, dtype=np.float32)
cv.imshow("image", image)
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
h, w = src.shape[:2]
for row in range(h):
    for col in range(w):
        dist = cv.pointPolygonTest(contours[0], (row, col), True)
        # True的话返回点到轮廓的距离，False则返回+1，0，-1三个值，其中+1表示点在轮廓内部，0表示点在轮廓上，-1表示点在轮廓外

        # 此功能可查找图像中的点与轮廓之间的最短距离.
        # 当点在轮廓外时返回负值，当点在内部时返回正值，如果点在轮廓上则返回零.
        # dist = cv2.pointPolygonTest（cnt，（50,50），True）
        # 在函数中，第一个参数，必须要是数组，第二个参数为要查找的点，第三个参数是measureDist。
        # 第三个参数，如果为True，则查找签名距离. 如果为False，则查找该点是在内部还是外部或在轮廓上（它分别返回+1，-1,0）

        if dist == 0:
            image[row, col] = (255, 255, 255)
        if dist > 0:
            image[row, col] = (255-dist, 0, 0)
        if dist < 0:
            image[row, col] = (255, 0, 255+dist)

dst = cv.convertScaleAbs(image)
# 该操作可实现图像增强等相等的操作
# 将像素点进行绝对值计算
# dst = np.uint8(dst)
# 显示
cv.imshow("contours_analysis", dst)
cv.imwrite("contours_analysis.png", dst)
cv.waitKey(0)
cv.destroyAllWindows()
