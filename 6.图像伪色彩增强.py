import cv2 as cv

src = cv.imread("./pictures/factory.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
dst = cv.applyColorMap(src, cv.COLORMAP_HSV)
cv.imshow("input", src)
cv.imshow("output", dst)

# 伪色彩
image = cv.imread("./pictures/robot.jpg")
color_image = cv.applyColorMap(image, cv.COLORMAP_JET)
cv.imshow("robot", image)
cv.imshow("color_robot", color_image)

cv.waitKey(0)
cv.destroyAllWindows()
