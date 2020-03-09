import cv2 as cv
import numpy as np


capture = cv.VideoCapture("./videos/Ora.mp4")
height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
count = capture.get(cv.CAP_PROP_FRAME_COUNT)
fps = capture.get(cv.CAP_PROP_FPS)
print(height, width, count, fps)
pt = []
lpt = [-1] * 8
Rpt = []
wring = " "
numThr = []

# 读取第一帧
ret, frame = capture.read()
cv.namedWindow("Ora_DEMO", cv.WINDOW_AUTOSIZE)
cv.putText(frame, "CHOSE ROI", (5, 68), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
# 可以在图片上选择roi区域
x, y, w, h = cv.selectROI("Ora_DEMO", frame, True, False)
track_window = (x, y, w, h)


def LineArea_Detection():
    a, b, c, d, e, f, g, h, i, j = [], [], [], [], [], [], [], [], [], []
    Xab, Xcd = 0, 0
    area = 6
    for k in range(len(lpt)):
        if k < 2:
            a.append(lpt[k])
        if 2 <= k < 4:
            b.append(lpt[k])
        if 4 <= k < 6:
            c.append(lpt[k])
        if 6 <= k < 8:
            d.append(lpt[k])
    for s in range(len(Rpt)):
        if s < 2:
            e.append(Rpt[s])
        if 2 <= s < 4:
            g.append(Rpt[s])
        if 4 <= s < 6:
            h.append(Rpt[s])
        if 6 <= s < 8:
            f.append(Rpt[s])
    for p in range(len(e)):
        i.append((e[p] + g[p]) / 2)
        j.append((f[p] + h[p]) / 2)
    print("a , b, c, d:", a, b, c, d)
    print("e , f, g, h:", e, f, g, h)
    print("i, j :", i, j)

    if a[0] is not -1:
        if a[0] > i[0]:
            Kab = (b[1] - a[1]) / (b[0] - a[0])
            Xa = (i[1] + Kab * a[0] - a[1]) / Kab
            Xai = abs((abs(i[0]) - abs(Xa)))
            Xb = (j[1] + Kab * a[0] - a[1]) / Kab
            Xbj = abs((abs(j[0]) - abs(Xb)))
            area = (Xai + Xbj) / 2
            numThr.append(area)
            print(area)
            print("numTreshold: ", numThr)
    # if c[0] & d[0] is not -1:
    #     Kcd = (d[1] - c[1]) / (d[0] - c[0])
    #     Xcd = (i[1] - Kcd * c[0] - c[1]) / Kcd
            area = abs(numThr[0] - numThr[len(numThr)-1])
            print("area 1:", area)
    return area


def Decision_r(area):
    e, f, g, h, i, j = [], [], [], [], [], []
    for s in range(len(Rpt)):
        if s < 2:
            e.append(Rpt[s])
        if 2 <= s < 4:
            g.append(Rpt[s])
        if 4 <= s < 6:
            h.append(Rpt[s])
        if 6 <= s < 8:
            f.append(Rpt[s])
    for p in range(len(e)):
        i.append((e[p] + g[p]) / 2)
        j.append((f[p] + h[p]) / 2)
    War = "Safe"
    widP = (abs(abs(g[0]) - abs(i[0])) / 2) + abs(abs(abs(h[0]) - abs(j[0])) / 2) /2
    Thr = abs(widP / 11.5)
    print("pidai thr : ", Thr)
    # if (Thr - (Thr / 4)) <= area <= (Thr + (Thr / 4)):
    #     War = War
    if area > Thr:
        War = "Dangerous"
    if area < Thr:
        War = "Safe"
    return War


def canny(image):
    t = 80
    canny_output = cv.Canny(image, t, t * 2)
    return canny_output


def line_Detection(src):
    binary = canny(src)
    linesP = cv.HoughLinesP(binary, 1, np.pi / 360, 245, None, 100, 350)
    print(linesP)
    # cv.HoughLinesP 霍夫变化的概论形式
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            print("L is :", l)
            if wring is "Safe":
                cv.line(src, (l[0], l[1]), (l[2], l[3]), (255, 125, 0), 5, cv.LINE_AA)
            if wring is "Dangerous":
                cv.line(src, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 5, cv.LINE_AA)
            for c in range(len(l)):
                lpt.insert(c, l[c])
                lpt.pop()
    return src


def selectROI_MUL(winname, pts, img):
    """

    :param winname: 用于显示图片的窗口
    :param pts: 用于保存选取点的list
    :param img: 需要选取ROI区域的图片
    """
    import cv2
    # import numpy as np
    cv2.namedWindow(winname)

    # 统一的：mouse callback function
    def draw_roi(event, x, y, flags, param):
        img2 = img.copy()

        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击，选择点
            pts.append((x, y))

        if event == cv2.EVENT_RBUTTONDOWN:  # 右键点击，取消最近一次选择的点
            pts.pop()

        if event == cv2.EVENT_MBUTTONDOWN:  # 中键绘制轮廓
            mask = np.zeros(img.shape, np.uint8)
            points = np.array(pts, np.int32)
            points = points.reshape((-1, 1, 2))
            # 画多边形
            mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
            mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))  # 用于求 ROI
            mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))  # 用于 显示在桌面的图像

            show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)

            cv2.imshow("mask", mask2)
            cv2.imshow("show_img", show_image)

            ROI = cv2.bitwise_and(mask2, img)
            cv2.imshow("ROI", ROI)
            cv2.waitKey(0)

        if len(pts) > 0:
            # 将pts中的最后一点画出来
            cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)

        if len(pts) > 1:
            # 画线
            for i in range(len(pts) - 1):
                cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)  # x ,y 为鼠标点击地方的坐标
                cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

        cv2.imshow(winname, img2)

    cv2.setMouseCallback(winname, draw_roi)
    print("[INFO] 单击左键：选择点，单击右键：删除上一次选择的点，单击中键：确定ROI区域")
    print("[INFO] 按 ENTER 退出")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            if len(pts) >= 3:
                break
            else:
                print('点数小于3，不能构成多边形，请继续选点')
    cv2.destroyAllWindows()


cv.putText(frame, "Ora Detection", (5, 28), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
cv.putText(frame, "ROI Region", (x + 22, y - 3), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
selectROI_MUL("Ora_DEMO", pt, frame)
roi = frame[y:y + h, x:x + w]  # 获取ROI直方图
result = line_Detection(roi)
for c in range(len(pt)):
    if len(Rpt) < len(pt) * len(pt[c]):
        for c1 in range(len(pt[c])):
            Rpt.append(pt[c][c1])
while True:
    ret, frame = capture.read()
    if ret is True:
        fps_1 = capture.get(cv.CAP_PROP_FPS)
        roi = frame[y:y + h, x:x + w]
        result = line_Detection(roi)
        if lpt is not None and len(lpt) == 8:
            wring = Decision_r(LineArea_Detection())
        for num_i in range(8):
            lpt.pop()
            lpt.insert(num_i, -1)
        print("lpt :", lpt)
        print("rpt :", Rpt)

        if wring is "Safe":
            cv.putText(frame, "Ora Detection", (5, 28), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv.putText(frame, "FPS: %d" % fps_1, (5, 58), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv.putText(frame, "Status: %s" % wring, (5, 88), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv.putText(frame, "ROI-Region:", (x + 12, y - 8), cv.FONT_HERSHEY_SIMPLEX
                       , 0.8, (255, 255, 0), 2)
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            cv.putText(frame, "Detection Region:", (pt[len(pt) - 1][0] + 12, pt[len(pt) - 1][1] + 18)
                       , cv.FONT_HERSHEY_SIMPLEX, 0.8, (15, 255, 255), 2)
            for c in range(len(pt)):
                if c < len(pt) - 1:
                    cv.line(frame, pt[c], pt[c + 1], (15, 255, 255), 2)
                if c == len(pt) - 1:
                    cv.line(frame, pt[c], pt[0], (15, 255, 255), 2)

        if wring == "Dangerous":
            cv.putText(frame, "Detection Region:", (pt[len(pt) - 1][0] + 12, pt[len(pt) - 1][1] + 18)
                       , cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv.putText(frame, "ROI-Region:", (x + 12, y - 8), cv.FONT_HERSHEY_SIMPLEX
                       , 0.8, (255, 255, 0), 2)
            for c in range(len(pt)):
                if c < len(pt) - 1:
                    cv.line(frame, pt[c], pt[c + 1], (0, 0, 255), 2)
                if c == len(pt) - 1:
                    cv.line(frame, pt[c], pt[0], (0, 0, 255), 2)

            cv.putText(frame, "Ora Detection", (5, 28), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv.putText(frame, "Status: %s" % wring, (5, 88), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv.putText(frame, "FPS: %d" % fps_1, (5, 58), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv.namedWindow("result", cv.WINDOW_FREERATIO)
        cv.imshow("result", frame)
        c = cv.waitKey(50)
        if c == 27:  # ESC
            break
    else:
        break
