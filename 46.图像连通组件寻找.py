import cv2 as cv
import numpy as np
# 查找图像的连通集


def connected_components_demo(src):
    src = cv.GaussianBlur(src, (3, 3), 0)  # 高斯模糊
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)  # 灰度值
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值化
    cv.imshow("binary", binary)
    cv.imwrite('binary.png', binary)

    output = cv.connectedComponents(binary, connectivity=8, ltype=cv.CV_32S)  # 连通域
    # def connectedComponents(image, labels=None, connectivity=None, ltype=None)
    # image ： 应该是8位的单通道图像
    # labels : 目标标签图像
    # connectivity： 可以选择为8路或4路
    # ltype： 输出图像标签类型。目前支持cv_s和cv_u
    print("type_output:", type(output))
    print(output)
    num_labels = output[0]
    print(num_labels)  # output: 5
    labels = output[1]

    # 构造颜色
    colors = []
    for i in range(num_labels):      # 循环数据函数
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        colors.append((b, g, r))
    colors[0] = (0, 0, 0)
    print("len type:", len(colors), type(colors))
    print(colors)

    # 画出连通图
    h, w = gray.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            image[row, col] = colors[labels[row, col]]

    cv.imshow("colored labels", image)
    cv.imwrite("labels.png", image)
    print("total componets : ", num_labels - 1)


src = cv.imread("./pictures/robot.jpg")
h, w = src.shape[:2]
connected_components_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()

