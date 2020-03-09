import cv2 as cv
import numpy as np
# 图像连通组件状态统


def connected_components_stats_demo(src):
    src = cv.GaussianBlur(src, (3, 3), 0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary_ = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # 使用开运算去掉外部的噪声
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    binary = cv.morphologyEx(binary_, cv.MORPH_OPEN, kernel)
    cv.imshow("binary", binary_)

    num_labels, labels, stats, centers = cv.connectedComponentsWithStats(binary, connectivity=8, ltype=cv.CV_32S)
    colors = []
    for i in range(num_labels):
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        colors.append((b, g, r))

    colors[0] = (0, 0, 0)
    image = np.copy(src)
    for t in range(1, num_labels, 1):
        x, y, w, h, area = stats[t]
        cx, cy = centers[t]
        # 标出中心位置
        cv.circle(image, (np.int32(cx), np.int32(cy)), 2, (0, 255, 0), 2, 8, 0)
        # 画出外接矩形
        cv.rectangle(image, (x, y), (x + w, y + h), colors[t], 1, 8, 0)
        cv.putText(image, "No." + str(t), (x, y), cv.FONT_HERSHEY_COMPLEX_SMALL, .6, (0, 0, 255), 1)
        # void cv::putText(
        # 		cv::Mat& img, // 待绘制的图像
        # 		const string& text, // 待绘制的文字
        # 		cv::Point origin, // 文本框的左下角
        # 		int fontFace, // 字体 (如cv::FONT_HERSHEY_PLAIN)
        # 		double fontScale, // 尺寸因子，值越大文字越大
        # 		cv::Scalar color, // 线条的颜色（RGB）
        # 		int thickness = 1, // 线条宽度
        # 		int lineType = 8, // 线型（4邻域或8邻域，默认8邻域）
        # 		bool bottomLeftOrigin = false // true='origin at lower left'
        # 	)
        print("label index %d, area of the label : %d" % (t, area))

    cv.imshow("colored labels", image)
    cv.imwrite("labels.png", image)
    print("total rice : ", num_labels - 1)


input = cv.imread("./pictures/robot.jpg")
connected_components_stats_demo(input)
cv.waitKey(0)
cv.destroyAllWindows()
