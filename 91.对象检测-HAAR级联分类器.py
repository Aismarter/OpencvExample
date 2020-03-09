import cv2 as cv

capture = cv.VideoCapture(0)
detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_alt.xml")  # 人脸检测
eye_detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")  # 眼睛检测
smile_detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_smile.xml")  # 微笑检测

# image = cv.imread('people.jpg')
# faces = detector.detectMultiScale(image, scaleFactor=1.05, minNeighbors=1,
#                                           minSize=(50, 50), maxSize=(500, 500))
# for x, y, width, height in faces:
#     cv.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 2, cv.LINE_8, 0)
# cv.imshow("faces", image)
# cv.waitKey(0)
# cv.imwrite('face.jpg', image)

while True:
    ret, image = capture.read()
    if ret is True:
        cv.imshow("frame", image)

        # 人脸检测
        faces = detector.detectMultiScale(image, scaleFactor=1.05, minNeighbors=1,
                                          minSize=(30, 30), maxSize=(500, 500))
        for x, y, width, height in faces:
            cv.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2, cv.LINE_8, 0)

        # 眼睛检测 -- 基于人脸
        # 提取人脸
        roi = image[y:y + height, x:x + width]
        # 检测人眼
        eyes = eye_detector.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5,
                                             minSize=(20, 20), maxSize=(30, 30))
        # 绘制人眼
        for ex, ey, ew, eh in eyes:
            cv.rectangle(roi, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        # 微笑检测
        smiles = smile_detector.detectMultiScale(roi, scaleFactor=1.05, minNeighbors=90,
                                                 minSize=(20, 20))
        # 绘制微笑框
        for sx, sy, sw, sh in smiles:
            cv.rectangle(roi, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            cv.putText(image, 'Smile', (x + 10, y - 7), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv.imshow("faces", image)
        c = cv.waitKey(50)
        if c == 27:
            break
    else:
        break
cv.destroyAllWindows()
