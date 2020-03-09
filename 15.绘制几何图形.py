import cv2 as cv
import numpy as np

image = np.zeros((512, 512, 3), dtype=np.uint8)

cv.rectangle(image, (100, 100), (300, 300), (125, 0, 0), 20, cv.LINE_8, 1)
# image 表示图像
# pt1 和 pt2 参数分别代表矩形的左上角和右下角两个点，
# color 参数一般用 RGB 值指定，表示矩形边框的颜色
# thickness 参数表示矩形边框的厚度，如果为负值，如 CV_FILLED，则表示填充整个矩形。
# 这个参数看上去是指定 Bresenham 算法是 4 连通的还是 8 连通的，涉及到了计算机图形学的知识。如果指定为 CV_AA，则是使用高斯滤波器画反锯齿线。
# shift 参数表示点坐标中的小数位数，但是我感觉这个参数是在将坐标右移 shift 位一样

cv.circle(image, (256, 256), 150, (0, 0, 115), 30, cv.LINE_8, 0)
# cvCircle(CvArr* img, CvPoint center, int radius, CvScalar color, int thickness=1, int lineType=8, int shift=0)
# img为源图像指针
# center为画圆的圆心坐标
# radius为圆的半径
# color为设定圆的颜色，规则根据B（蓝）G（绿）R（红）
# thickness 如果是正数，表示组成圆的线条的粗细程度。否则，表示圆是否被填充
# line_type 线条的类型。默认是8
# shift 圆心坐标点和半径值的小数点位数

cv.ellipse(image, (256, 256), (150, 50), 360, 0, 360, (0, 255, 0), 2, cv.LINE_8, 0)
# 图像。
# 椭圆的中心。
# 轴椭圆主轴尺寸的一半。
# 角度椭圆旋转角度（度）。
# 椭圆弧的起始角（度）。
# 椭圆弧的结束角（度）。
# 颜色椭圆颜色。
# 椭圆弧轮廓的厚度，如果为正。否则，这表明将绘制一个填充椭圆扇区。
# 椭圆边界的线型类型。参见线型
# 偏移中心坐标和轴值中的小数位数
cv.imshow("image", image)
cv.waitKey(0)
# cvWaitKey()函数的功能是不断刷新图像，频率时间为delay，单位为ms。
# delay>0时，延迟”delay”ms，在显示视频时这个函数是有用的，用于设置在显示完一帧图像后程序等待”delay”ms再显示下一帧视频；
# 如果使用cvWaitKey(0)则只会显示第一帧视频。
#
# 返回值：如果delay>0,那么超过指定时间则返回-1；如果delay=0，将没有返回值。
#
# 如果程序想响应某个按键，可利用if(cvWaitKey(1)==Keyvalue)；
# 经常程序里面出现if( cvWaitKey(10) >= 0 ) 是说10ms中按任意键进入此if块。
#
# 注意：在imshow之后如果没有waitKey语句则不会正常显示图像。

for i in range(100000):
    image[:, :, :] = 0
    x1 = np.random.rand() * 512
    y1 = np.random.rand() * 512
    x2 = np.random.rand() * 512
    y2 = np.random.rand() * 512

    b = np.random.rand(0, 256)
    g = np.random.rand(0, 256)
    r = np.random.rand(0, 256)
    cv.line(image, (np.int(x1), np.int(y1)), (np.int(x2), np.int(y2)), (b, g, r), 4, cv.LINE_8, 0)
    # 　　第一个参数 img：要划的线所在的图像;
    # 　　第二个参数 pt1：直线起点
    # 　　第三个参数 pt2：直线终点
    # 　　第四个参数 color：直线的颜色
    # 　　第五个参数 thickness=1：线条粗细

    # cv.rectangle(image, (np.int(x1), np.int(y1)), (np.int(x2), np.int(y2)), (b, g, r), 1, cv.LINE_8, 0)
    cv.imshow("image", image)
    c = cv.waitKey(20)
    if c == 27:
        break

cv.imshow("image", image)
cv.waitKey(0)
cv.destroyAllWindows()

