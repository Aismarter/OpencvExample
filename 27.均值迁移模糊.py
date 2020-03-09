# 均值迁移模糊
# meanShfit均值漂移（均值迁移）算法是一种通用的聚类算法，
# 它的基本原理是：对于给定的一定数量样本，任选其中一个样本，
# 以该样本为中心点划定一个圆形区域，求取该圆形区域内样本的质心，即密度最大处的点，
# 再以该点为中心继续执行上述迭代过程，直至最终收敛。
# 可以利用均值偏移算法的这个特性，实现彩色图像分割，Opencv中对应的函数是pyrMeanShiftFiltering。
# 这个函数严格来说并不是图像的分割，而是图像在色彩层面的平滑滤波，
# 它可以中和色彩分布相近的颜色，平滑色彩细节，侵蚀掉面积较小的颜色区域，
# 所以在Opencv中它的后缀是滤波“Filter”，而不是分割“segment”。

import cv2 as cv
import numpy as np

capture = cv.VideoCapture("./videos/Ora.mp4")
height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
count = capture.get(cv.CAP_PROP_FRAME_COUNT)
fps = capture.get(cv.CAP_PROP_FPS)
print(height, width, count, fps)


def MeanShift_process(src):
    dst = cv.pyrMeanShiftFiltering(src, 30, 50, termcrit=(cv.TERM_CRITERIA_MAX_ITER +
                                                          cv.TERM_CRITERIA_EPS, 5, 1))
    # 函数原型
    # void pyrMeanShiftFiltering( InputArray src, OutputArray dst,
    #                                          double sp, double sr, int maxLevel=1,
    #                                          TermCriteria termcrit=TermCriteria(
    #                                             TermCriteria::MAX_ITER+TermCriteria::EPS,5,1) );
    # 第一个参数src，输入图像，8位，三通道的彩色图像，并不要求必须是RGB格式，HSV、YUV等Opencv中的彩色图像格式均可；
    # 第二个参数dst，输出图像，跟输入src有同样的大小和数据格式；
    # 第三个参数sp，定义的漂移物理空间半径大小；
    # 第四个参数sr，定义的漂移色彩空间半径大小；
    # 第五个参数maxLevel，定义金字塔的最大层数；
    # 第六个参数termcrit，定义的漂移迭代终止条件，可以设置为迭代次数满足终止，迭代目标与中心点偏差满足终止，或者两者的结合；
    cv.imshow("result", dst)


while True:
    ret, frame = capture.read()
    if ret is True:
        cv.imshow("input", frame)
        MeanShift_process(frame)
        c = cv.waitKey(0)
        if c == 27:  # esc
            break
    else:
        break


