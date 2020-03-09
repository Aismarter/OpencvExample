# 识别与跟踪视频中的特定颜色对象
# 这个是其实图像处理与二值分析的视频版本，通过读取视频每一帧的图像，
# 然后对图像二值分析，得到指定的色块区域，主要步骤如下：
#
# 色彩转换BGR2HSV
# inRange提取颜色区域mask
# 对mask区域进行二值分析得到位置与轮廓信息
# 绘制外接椭圆与中心位置
# 显示结果
# 其中涉及到的知识点主要包括图像处理、色彩空间转换、形态学、轮廓分析等
import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)
height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
count = capture.get(cv.CAP_PROP_FRAME_COUNT)
fps = capture.get(cv.CAP_PROP_FPS)
print(height, width, count, fps)
# ./videos/
out = cv.VideoWriter("scratch_color.avi", cv.VideoWriter_fourcc('D', 'I', 'V', 'X'), 15,
                     (np.int(width), np.int(height)), True)


def process(image, opt=1):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    line = cv.getStructuringElement(cv.MORPH_CROSS, (15, 15), (-1, -1))
    # getStructuringElement函数会返回指定形状和尺寸的结构元素。
    #     # 这个函数的第一个参数表示内核的形状，有三种形状可以选择。
    #     # 矩形：MORPH_RECT;
    #     # 交叉形：MORPH_CROSS;
    #     # 椭圆形：MORPH_ELLIPSE;
    # 第二和第三个参数分别是内核的尺寸以及锚点的位置。
    # 一般在调用erode以及dilate函数之前，先定义一个Mat类型的变量来获得getStructuringElement函数的返回值。
    # 对于锚点的位置，有默认值Point（-1,-1），表示锚点位于中心点。
    # element形状唯一依赖锚点位置，其他情况下，锚点只是影响了形态学运算结果的偏移。

    mask = cv.inRange(hsv,  (0, 43, 48), (65, 255, 255))
    cv.imshow("mask", mask)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, line)
    cv.imshow("masks", mask)

    # 轮廓提取, 发现最大轮廓
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    index = -1
    max = 0
    for c in range(len(contours)):
        area = cv.contourArea(contours[c])
        if area > max:
            max = area
            index = c
    # 绘制
    if index >= 0:
        rect = cv.minAreaRect(contours[index])
        cv.ellipse(image, rect, (255, 255, 0), 2, 8)
        cv.circle(image, (np.int32(rect[0][0]), np.int32(rect[0][1])), 2, (255, 255, 0), 2, 8, 0)
        x, y, w, h = cv.boundingRect(contours[index])
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # for c in range(len(contours)):
    #     rect = cv.minAreaRect(contours[c])
    #     cv.ellipse(image, rect, (255, 255, 0), 2, 8)
    #     cv.circle(image, (np.int32(rect[0][0]), np.int32(rect[0][1])), 2, (255, 255, 0), 2, 8, 0)
    #     x, y, w, h = cv.boundingRect(contours[c])
    #     cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)
    return image


while True:

    ret, frame = capture.read()
    if ret is True:
        cv.imshow("video-input", frame)
        result = process(frame)
        cv.imshow("result", result)
        out.write(result)
        c = cv.waitKey(50)
        # print(c)
        if c == 27:  # ESC
            cv.imwrite("result.jpg", result)
            break
    else:
        break

cv.waitKey(0)
cv.destroyAllWindows()

capture.release()
out.release()


