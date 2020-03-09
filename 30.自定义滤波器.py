import cv2 as cv
import numpy as np

src = cv.imread("./pictures/robot.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

blur_op = np.ones([5, 5], dtype=np.float32)/25.
shape_op = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]], np.float32)
grad_op = np.array([[1, 0], [0, -1]], dtype=np.float32)

dst1 = cv.filter2D(src, -1, blur_op)
dst2 = cv.filter2D(src, -1, shape_op)
dst3 = cv.filter2D(src, cv.CV_32F, grad_op)

cv.imshow("blur_op", dst1)
cv.imshow("shape_op", dst2)  # 类似于锐化的效果
cv.imshow("grad_op"
          "", dst3)

cv.waitKey(0)
cv.destroyAllWindows()

