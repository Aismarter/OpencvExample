import numpy as np
import cv2


def process(image, opt=1):
    # Detecting corners
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 35, 0.1, 100)
    # cv::goodFeaturesToTrack()，它不仅支持Harris角点检测，也支持Shi Tomasi算法的角点检测。
    # 但是，该函数检测到的角点依然是像素级别的，
    # 若想获取更为精细的角点坐标，则需要调用cv::cornerSubPix()函数进一步细化处理，即亚像素。
    # 函数原型void cv::goodFeaturesToTrack(
    # 	cv::InputArray image, // 输入图像（CV_8UC1 CV_32FC1）
    # 	cv::OutputArray corners, // 输出角点vector
    # 	int maxCorners, // 最大角点数目
    # 	double qualityLevel, // 质量水平系数（小于1.0的正数，一般在0.01-0.1之间）
    # 	double minDistance, // 最小距离，小于此距离的点忽略
    # 	cv::InputArray mask = noArray(), // mask=0的点忽略
    # 	int blockSize = 3, // 使用的邻域数
    # 	bool useHarrisDetector = false, // false ='Shi Tomasi metric'
    # 	double k = 0.04 // Harris角点检测时使用
    # )
    print(len(corners))
    for pt in corners:
        print(pt)
        # b = np.random.random_integers(0, 256)
        # g = np.random.random_integers(0, 256)
        # r = np.random.random_integers(0, 256)
        x = np.int32(pt[0][0])
        y = np.int32(pt[0][1])
        # cv2.circle(image, (x, y), 5, (int(b), int(g), int(r)), 2)
        cv2.circle(image, (x, y), 5, (0, 255, 0), 2)
    # output
    return image


src = cv2.imread("./pictures/robot.jpg")
cv2.imshow("input", src)
result = process(src)
cv2.imshow('result', result)
cv2.imwrite('result.jpg', result)
cv2.waitKey(0)
