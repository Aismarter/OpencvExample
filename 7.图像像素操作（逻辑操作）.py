import cv2 as cv
import numpy as np

# create  image one
src1 = np.zeros(shape=[400, 400, 3], dtype=np.uint8)
src1[100:200, 100:200, 1] = 255
src1[100:200, 100:200, 2] = 255
cv.imshow("input1", src1)

# create image two
src2 = np.zeros(shape=[400, 400, 3], dtype=np.uint8)
src2[150:250, 150:250, 2] = 255
cv.imshow("input2", src2)

dst1 = cv.bitwise_and(src1, src2)
dst2 = cv.bitwise_or(src1, src2)
dst3 = cv.bitwise_xor(src1, src2)
dst4 = cv.bitwise_not(src1, src2)

cv.imshow("dst1", dst1)
cv.imshow("dst2", dst2)
cv.imshow("dst3", dst3)
cv.imshow("dst4", dst4)

cv.waitKey(0)
cv.destroyAllWindows()

