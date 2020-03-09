import cv2
# 图片读取与显示
src = cv2.imread("./pictures/factory.jpg")
cv2.namedWindow("input", cv2.WINDOW_FREERATIO)
cv2.imshow("input", src)
cv2.waitKey(0)
cv2.destroyAllWindows()

