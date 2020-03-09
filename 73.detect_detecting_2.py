import cv2 as cv
import numpy as np


def get_template(binary, boxes):  # 获取模板
    x, y, w, h = boxes[0]
    roi = binary[y:y + h, x:x + w]
    return roi


def detect_defect(binary, boxes, tpl):  # 缺陷函数检测
    height, width = tpl.shape
    index = 1
    defect_rois = []
    # 发现缺失
    for x, y, w, h in boxes:
        roi = binary[y:y + h, x:x + w]
        roi = cv.resize(roi, (width, height))
        # cv.namedWindow("roi", cv.WINDOW_FREERATIO)
        cv.imshow("roi", roi)
        mask = cv.subtract(tpl, roi)  # 图像减法

        # 根据模板进行相减得到与模板不同的区域
        cv.namedWindow("mask", cv.WINDOW_FREERATIO)
        cv.imshow("mask", mask)
        se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5), (-1, -1))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, se)

        # 对这些区域进行形态学操作，去掉边缘细微差异
        cv.namedWindow("mask1", cv.WINDOW_FREERATIO)
        cv.imshow("mask1", mask)
        ret, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
        cv.namedWindow("mask2", cv.WINDOW_FREERATIO)
        cv.imshow("mask2", mask)
        count = 0
        for row in range(height):
            for col in range(width):
                pv = mask[row, col]
                if pv == 255:
                    count += 1
        if count > 0:
            defect_rois.append([x, y, w, h])  # 检测缺陷的掩码图样并将其相加
        cv.imwrite("mask%d.png" % index, mask)
        index += 1
        # 最终就得到了可以检出的缺陷或者划痕刀片
    return defect_rois


src = cv.imread("./pictures/ce_02.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

# 图像二值化
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))
# 这个函数的第一个参数表示内核的形状，有三种形状可以选择
# 矩  形： MORPH_RECT;
# 交叉形： MORPH_CROSS;
# 椭圆形： MORPH_ELLIPSE;
# 第二和第三个参数分别是内核的尺寸以及锚点的位置
binary = cv.morphologyEx(binary, cv.MORPH_OPEN, se)
cv.imshow("binary", binary)

# 轮廓提取
contours, hierarchy = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# 检测数据集
height, width = src.shape[:2]
rects = []
for c in range(len(contours)):  # 打出所有的点
    x, y, w, h = cv.boundingRect(contours[c])
    # x, y 是矩阵左上点的坐标， w, h 是矩阵的宽和高
    area = cv.contourArea(contours[c])
    if h > (height // 2):
        continue
    if area < 150:
        continue
    rects.append([x, y, w, h])  # 得到符合要求的点

# 对于得到的刀片外接矩形，首先需要通过排序，确定他们的编号.
# 排序轮廓
print(rects)
rects = sorted(rects, key=lambda x:x[1])  # 使用lambda函数指定以x的某些一个数据作为排序列表
# sorted() 函数与 reversed() 函数类似，该函数接收一个可迭代对象作为参数，返回一个对元素排序的列表。
# 在调用 sorted() 函数时，还可传入一个 key 参数，该参数可指定一个函数来生成排序的关键值。
# lambda的一般形式是关键字lambda后面跟一个或多个参数，紧跟一个冒号，以后是一个表达式。
# lambda是一个表达式而不是一个语句。它能够出现在Python语法不允许def出现的地方。
# 作为表达式，lambda返回一个值（即一个新的函数）。
# lambda用来编写简单的函数，而def用来处理更强大的任务。
# 一般的形式：
# f = lambda x, y, z :x+y+z
# 　　　　print f(1,2,3)  # 6
print(len(rects))
print(rects)
template = get_template(binary, rects)  # 获取模板

# 填充边缘
for c in range(len(contours)):
    x, y, w, h = cv.boundingRect(contours[c])
    area = cv.contourArea(contours[c])
    if h > (height // 2):
        continue
    if area < 150:
        continue
    cv.drawContours(binary, contours, c, 0, 2, 8)
    cv.imshow("binary-temple %d"%c, binary)
cv.namedWindow("template", cv.WINDOW_FREERATIO)
cv.imshow("template", template)

# 检测缺陷
defect_boxes = detect_defect(binary, rects, template)
for dx, dy, dw, dh in defect_boxes:
    cv.rectangle(src, (dx, dy), (dx + dw, dy + dh), (0, 0, 255), 1, 8, 0)
    cv.putText(src, "bad", (dx, dy - 2), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

index = 1
for dx, dy, dw, dh in rects:
    cv.putText(src, "num:%d" % index, (dx - 52, dy + 30), cv.FONT_HERSHEY_SIMPLEX, .5, (30, 122, 233), 2)
    index += 1

cv.imshow("result", src)
cv.imwrite("binary2.png", src)

cv.waitKey(0)
cv.destroyAllWindows()
