import cv2 as cv
import numpy as np

src = cv.imread("./pictures/st_02.png")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

# 图像二值化
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))
binary = cv.morphologyEx(binary, cv.MORPH_OPEN, se)
cv.imshow("binary", binary)

# 轮廓提取, 发现最大轮廓
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 寻找最大面积轮廓
cnt_maxArea = sorted(contours, key=cv.contourArea)[0]

# 寻找最小外接矩形,返回最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
rect = cv.minAreaRect(cnt_maxArea)
print(rect[2])
print(rect[0])
# trick
height, width = rect[1]
print(rect[1])
box = cv.boxPoints(rect)
src_pts = np.int0(box)
print(src_pts)

dst_pts = []
dst_pts.append([width, height])
dst_pts.append([0, height])
dst_pts.append([0, 0])
dst_pts.append([width, 0])

# 透视变换
M, status = cv.findHomography(src_pts, np.array(dst_pts))  # 原图到透视后的图的四个点的转换矩阵
# findHomography() 函数。 功能： 找到两个平面之间的转换矩阵。
# src_pts 源平面中点的坐标矩阵
# dst_ptd 目标平面中点的坐标矩阵

result = cv.warpPerspective(src, M, (np.int32(width), np.int32(height)))
# warpPerspective 透视变换函数
# 透视变换（perspective transformation） 是将图像投影到一个新的视平面，也称为投射映射

if height < width:
    result = cv.rotate(result, cv.ROTATE_90_CLOCKWISE)  # rotate 旋转

cv.imshow("result", result)

cv.waitKey(0)
cv.destroyAllWindows()
