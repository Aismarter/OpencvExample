import cv2 as cv
import numpy as np


def canny_demo(image):
    t = 80
    canny_output = cv.Canny(image, t, t * 2)
    cv.imshow("canny_output", canny_output)
    cv.imwrite("canny_output.png", canny_output)
    return canny_output


src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
binary = canny_demo(src)
k = np.ones((3, 3), dtype=np.uint8)
binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)

# 轮廓发现
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for c in range(len(contours)):
    # 椭圆拟合
    if len(contours[c]) >= 5:
        (cx, cy), (a, b), angle = cv.fitEllipse(contours[c])  # 拟合椭圆函数 这里必须要输入参数大于5个

        # 绘制椭圆
        cv.ellipse(src, (np.int32(cx), np.int32(cy)),
                   (np.int32(a/2), np.int32(b/2)), angle, 0, 360, (0, 0, 255), 2, 8, 0)
        # void cvEllipse( CvArr* img, CvPoint center, CvSize axes, double angle,
        #                 double start_angle, double end_angle, CvScalar color,
        #                 int thickness=1, int line_type=8, int shift=0 );
        # img 图像。
        # center 椭圆圆心坐标。
        # axes 轴的长度。
        # angle 偏转的角度。
        # start_angle 圆弧起始角的角度。.
        # end_angle 圆弧终结角的角度。
        # color 线条的颜色。
        # thickness 线条的粗细程度。
        # line_type 线条的类型,见CVLINE的描述。
        # shift 圆心坐标点和数轴的精度。


# 显示
cv.imshow("contours_analysis", src)
cv.imwrite("contours_analysis.png", src)
cv.waitKey(0)
cv.destroyAllWindows()

