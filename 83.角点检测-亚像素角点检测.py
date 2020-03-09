import numpy as np
import cv2 as cv


def process(image, opt=1):
    # Detecting corners
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, 35, 0.3, 50)
    print(len(corners))
    for pt in corners:
        print(pt)
        x = np.int32(pt[0][0])
        y = np.int32(pt[0][1])
        cv.circle(image, (x, y), 5, (0, 255, 0), 2)

    # detect sub-pixel
    winSize = (3, 3)
    zeroZone = (-1, -1)

    #  Stop condition
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)

    # Calculate the refined corner location
    corners = cv.cornerSubPix(gray, corners, winSize, zeroZone, criteria)

    # display
    for i in range(corners.shape[0]):
        print("-- Refind Corner [", i, "] (", corners[i, 0, 0], ",", corners[i, 0, 1], ")")
    return image


# src = cv.imread("./pictures/robot.jpg")
# cv.imshow("input", src)
# result = process(src)
# cv.imshow("result", result)
#
# cv.waitKey(0)
# cv.destroyAllWindows()

cap = cv.VideoCapture("./videos/Ora.mp4")
while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    cv.imshow("input", frame)
    result = process(frame)
    cv.imshow("result", result)
    k = cv.waitKey(50) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
