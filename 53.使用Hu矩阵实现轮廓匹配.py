import cv2 as cv
import numpy as np


def contours_info(image):
    # openCV自带寻找轮廓的函数，流程是：获取灰度图-> 图片二值化 -> 寻找轮廓
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours，定义为“vector<vector<Point>> contours”，是一个向量，并且是一个双重向量，向量
    #            内每个元素保存了一组由连续的Point点构成的点的集合的向量，每一组Point点集就是一个轮廓。
    #            有多少轮廓，向量contours就有多少元素。
    # hierarchy向量内每一个元素的4个int型变量——hierarchy[i][0] ~hierarchy[i][3]，分别表示第
    #         i个轮廓的后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号。如果当前轮廓没有对应的后一个
    #         轮廓、前一个轮廓、父轮廓或内嵌轮廓的话，则hierarchy[i][0] ~hierarchy[i][3]的相应位被设置为
    #         默认值-1。
    # 第四个参数：int型的mode，定义轮廓的检索模式：
    #   取值一：CV_RETR_EXTERNAL只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略
    #   取值二：CV_RETR_LIST   检测所有的轮廓，包括内围、外围轮廓，
    #           但是检测到的轮廓不建立等级关系，彼此之间独立，没有等级关系，
    #           这就意味着这个检索模式下不存在父轮廓或内嵌轮廓，
    #           所以hierarchy向量内所有元素的第3、第4个分量都会被置为-1，
    #   取值三：CV_RETR_CCOMP  检测所有的轮廓，但所有轮廓只建立两个等级关系，外围为顶层，
    #          若外围内的内围轮廓还包含了其他的轮廓信息，则内围内的所有轮廓均归属于顶层
    #   取值四：CV_RETR_TREE， 检测所有轮廓，所有轮廓建立一个等级树结构。
    #          外层轮廓包含内层轮廓，内层轮廓还可以继续包含内嵌轮廓。
    # 第五个参数：int型的method，定义轮廓的近似方法：
    #   取值一：CV_CHAIN_APPROX_NONE 保存物体边界上所有连续的轮廓点到contours向量内
    #   取值二：CV_CHAIN_APPROX_SIMPLE 仅保存轮廓的拐点信息，
    #         把所有轮廓拐点处的点保存入contours向量内，拐点与拐点之间直线段上的信息点不予保留
    #   取值三和四：CV_CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
    return contours


src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input1", cv.WINDOW_AUTOSIZE)
cv.imshow("input1", src)
src2 = cv.imread("./pictures/factory.jpg")
cv.imshow("input2", src2)

# 轮廓发现
contours1 = contours_info(src)
contours2 = contours_info(src2)

# 几何矩计算与hu矩计算
# opencv中提供了moments()来计算图像中的中心矩(最高到三阶)，
# HuMoments()用于由中心矩计算Hu矩.
# 同时配合函数contourArea函数计算轮廓面积和arcLength来计算轮廓或曲线长度
mm2 = cv.moments(contours2[0])
hum2 = cv.HuMoments(mm2)

# 轮廓匹配
for c in range(len(contours1)):
    mm = cv.moments(contours1[c])
    hum = cv.HuMoments(mm)
    dist = cv.matchShapes(hum, hum2, cv.CONTOURS_MATCH_I1, 0)
    # matchShapes() 函数作用是比较两个形状
    # double cvMatchShapes( const void* object1, const void* object2,
    #                       int method, double parameter=0 );
    # object1  第一个轮廓或灰度图像
    # object2  第二个轮廓或灰度图像
    # method  比较方法，其中之一 CV_CONTOUR_MATCH_I1, CV_CONTOURS_MATCH_I2 or CV_CONTOURS_MATCH_I3
    # parameter  比较方法的参数 (目前不用)
    if dist < 1:
        cv.drawContours(src, contours1, c, (0, 0, 255), 2, 8)
    print("dist %f" % (dist))

    mm1 = cv.moments(contours2[c])
    hum1 = cv.HuMoments(mm1)
    dist1 = cv.matchShapes(hum1, hum2, cv.CONTOURS_MATCH_I1, 0)
    if dist1 < 1:
        cv.drawContours(src2, contours2, c, (0, 0, 255), 2, 8)
    print("dist %f" % (dist1))

# 显示
cv.imshow("contours_analysis", src)
cv.imwrite("contours_analysis.png", src)
cv.imshow("contours_analysis1", src2)
cv.imwrite("contours_analysis1.png", src2)
cv.waitKey(0)
cv.destroyAllWindows()