import cv2 as cv
import numpy as np
# 膨胀会使图像边界不明显
# 腐蚀会使图像颜色深的地方更加明显

src = cv.imread("./pictures/factory.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

# 使用3x3结构元素进行膨胀与腐蚀操作
se = np.ones((3, 3), dtype=np.uint8)
# 你也可以创建任意一个二进制模板（binary mask）图像作为结构元素。
dilate = cv.dilate(src, se, None, (-1, -1), 1)  # 膨胀
"""
    函数原型 
    dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst
    使用特定的结构元素膨胀图像.
    该函数使用指定的结构元素来扩展源图像，该结构元素确定最大值被拍摄的像素邻域的形状。
    \f[\texttt{dst} (x,y) =  \max _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\f] 
    该函数支持in-place模式。扩张可以应用多次（迭代）, 对于多通道图像，每个通道都是独立处理的。
     src 输入图像; 可以是以下格式： CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
     dst 输出图像： 图像大小与输入图像相同.
     kernel 用于膨胀的结构元素; if elemenat=Mat(), a 3 x 3 的矩形元素结构被使用.
        Kernel 可以被 #getStructuringElement 元素创建
     anchor 表示锚在构件内的位置; 默认值 (-1, -1) 意味着锚点在中心位置
     iterations 使用膨胀操作的次数
     borderType 像素外推法
     borderValue border 固定边框时的值
    """
erode = cv.erode(src, se, None, (-1, -1), 1)  # 腐蚀
# 函数可以对输入图像用特定结构元素进行腐蚀操作，
# 该结构元素确定腐蚀操作过程中的邻域的形状，各点像素值将被替换为对应邻域上的最小值

# 显示
cv.imshow("dilate", dilate)
cv.imshow("erode", erode)
cv.imwrite("dilate.png", dilate)
cv.imwrite("erode.png", erode)
cv.waitKey(0)
cv.destroyAllWindows()

