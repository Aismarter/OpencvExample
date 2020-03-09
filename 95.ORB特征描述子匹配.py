import cv2 as cv

box = cv.imread("./pictures/box.png", 0)
box_in_sence = cv.imread("./pictures/box_in_scene.png", 0)
cv.imshow("box", box)
cv.imshow("box_in_sence", box_in_sence)

# 创建ORB特征检测器
orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(box, None)
kp2, des2 = orb.detectAndCompute(box_in_sence, None)
# 暴力匹配 汉明距离匹配特征点
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
# 绘制匹配
result = cv.drawMatches(box, kp1, box_in_sence, kp2, matches, None)
cv.imshow("orb-match", result)

# KNN match 特征检测器
matches_KNN = bf.knnMatch(des1, des2, k=1)
# 删除matches里面的空list，并且根据距离排序
while [] in matches_KNN:
    matches_KNN.remove([])
matches_KNN = sorted(matches_KNN, key=lambda x: x[0].distance)
# 画出最短距离的前15个点
result_KNN = cv.drawMatchesKnn(box, kp1, box_in_sence, kp2, matches_KNN[0:15], None, matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 255))
cv.imshow("orb-match-KNN", result_KNN)

# # 创建SIFT特征检测器
# orb = cv.xfeatures2d.SIFT_create()
# # SIFT 特征检测器 参数设置
# index_params = dict(algorithm=0, trees=5)
# search_params = dict(checks=20)
# flann = cv.FlannBasedMatcher(index_params, search_params)
# matches_SIFT = flann.knnMatch(des1, des2, k=2)
# # 记录好的点
# goodMatches = [[0, 0] for i in range(len(matches_SIFT))]
# for i, (m, n) in enumerate(matches_SIFT):
#     if m.distance < 0.7 * n.distance:
#         goodMatches[i] = [1, 0]
# draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=goodMatches, flags=0)
# result_SIFT = cv.drawMatchesKnn(box, kp1, box_in_sence, kp2, matches_SIFT, None, **draw_params)
# cv.imshow("orb-match-result_SIFT", result_SIFT)

cv.waitKey(0)
cv.destroyAllWindows()
