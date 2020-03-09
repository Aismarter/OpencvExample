import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)

height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
count = capture.get(cv.CAP_PROP_FRAME_COUNT)
fps = capture.get(cv.CAP_PROP_FPS)
print(height, width, count, fps)
out = cv.VideoWriter("./videos/test_camera.avi", cv.VideoWriter_fourcc('D', 'I', 'V', 'X'), 15,
                      (np.int(width), np.int(height)), True)
while True:
    ret, frame = capture.read()
    # cap.read()按帧读取视频
    # ret,frame是获cap.read()方法的两个返回值。
    # 其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。
    # frame就是每一帧的图像，是个三维矩阵。
    if ret is True:
        cv.imshow("video-input", frame)
        out.write(frame)
        c = cv.waitKey(50)
        if c == 27:
            break
    else:
        break

capture.release()
out.release()



