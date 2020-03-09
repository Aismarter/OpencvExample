import cv2 as cv
import numpy as np


def canny_demo(image):
    t = 80
    canny_output = cv.Canny(image, t, t * 2)  # Canny算法
    cv.imshow("canny_output", canny_output)
    cv.imwrite("canny_output.png", canny_output)
    return canny_output


src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
binary = canny_demo(src)
k = np.ones((3, 3), dtype=np.uint8)  # ones()返回一个全1的n维数组 作为处理函数的卷积核
binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)
# dst=cv.morphologyEx(src,op,kernel[, st[, anchor[, iterations[, borderType[, borderValue]]]]])
# src 图片资源， 图片的通道可以是任意的
# dst 输出图像， 要求原图像与目标图像相同
# op	形态操作函数
# kernel	卷积核

# 轮廓发现
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for c in range(len(contours)):
    # x, y, w, h = cv.boundingRect(contours[c])
    cv.drawContours(src, contours, c, (0, 0, 255), 1, 8)
    # cv.rectangle(src, (x, y), (x+w, y+h), (0, 0, 255), 1, 8, 0)

    # area = cv.contourArea(contours[c])  # 计算面积
    # arclen = cv.arcLength(contours[c], True)  # 计算弧长， True表示闭合区域
    # if area < 100 or arclen < 100:
    #     continue
    #
    # rect = cv.minAreaRect(contours[c])   # 返回包覆输入信息的最小斜矩形
    # cx, cy = rect[0]
    # box = cv.boxPoints(rect)  # 根据minAreaRect的返回值计算矩形的四个点
    # box = np.int0(box)
    # cv.drawContours(src, [box], 0, (0, 0, 255), 1)  # 画出轮廓图
    # cv.circle(src, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), 1, 8, 0)

# 显示
cv.namedWindow("contours_analysis", cv.WINDOW_FREERATIO)
cv.imshow("contours_analysis", src)
cv.imwrite("contours_analysis.png", src)
cv.waitKey(0)
cv.destroyAllWindows()
