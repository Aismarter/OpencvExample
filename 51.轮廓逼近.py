import cv2 as cv
import numpy as np

src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imshow("binary", binary)


# 轮廓发现
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print(type(contours), type(hierarchy))
for c in range(len(contours)):
    rect = cv.minAreaRect(contours[c])  # 返回一个包含输入信息的最小斜矩形
    x, y, w, h = cv.boundingRect(contours[c])
    cv.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 1, 8, 0)
    cv.drawContours(src, contours, c, (0, 0, 255), 1, 8)

    cx, cy = rect[0]
    box = cv.boxPoints(rect)  # 根据minAreaRect的返回值计算矩形的四个点
    result = cv.approxPolyDP(contours[c], 10, True)
    # approxPolyDP 主要功能是把一个连续光滑曲线折线化，对图像轮廓点进行多边形拟合。
    # InputArray curve:一般是由图像的轮廓点组成的点集
    # OutputArray approxCurve：表示输出的多边形点集
    # double epsilon：主要表示输出的精度，就是另个轮廓点之间最大距离数，5,6,7，，8，，,,，
    # bool closed：表示输出的多边形是否封闭
    vertexes = result.shape[0]
    if vertexes == 3:
        cv.putText(src, "triangle", (np.int32(cx), np.int32(cy)),
                   cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, 8)
    if vertexes == 4:
        cv.putText(src, "rectangle", (np.int32(cx), np.int32(cy)),
                   cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, 8)
    if vertexes == 6:
        cv.putText(src, "poly", (np.int32(cx), np.int32(cy)),
                   cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, 8)
    if vertexes > 10:
        cv.putText(src, "circle", (np.int32(cx), np.int32(cy)),
                   cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, 8)
    print(vertexes)


# 显示
cv.imshow("contours_analysis", src)
cv.imwrite("contours_analysis.png", src)
cv.waitKey(0)
cv.destroyAllWindows()
