# 图像插值 （图像插值用与缩小以及放大图片）
import cv2 as cv

src = cv.imread("./pictures/factory.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

h, w = src.shape[:2]
# 将图片高和宽分别赋值给x，y
print(h, w)

dst = cv.resize(src, (w*2, h*2), fx=1.75, fy=1.75, interpolation=cv.INTER_NEAREST)
# 将图像输出格式的长与宽放大到原来的两倍  沿沿x轴，y轴的缩放系数0.75  插入方式 最近邻插值
# cv2.resize(InputArray src, OutputArray dst, Size, fx, fy, interpolation)
# InputArray src	输入图片
# OutputArray dst	输出图片
# Size	输出图片尺寸
# fx, fy	沿x轴，y轴的缩放系数
# interpolation	插入方式
# INTER_NEAREST
# 最近邻插值
# INTER_LINEAR
# 双线性插值（默认设置）
# INTER_AREA
# 使用像素区域关系进行重采样。
# INTER_CUBIC
# 4x4像素邻域的双三次插值
# INTER_LANCZOS4
# 8x8像素邻域的Lanczos插值
cv.imshow("INTER_NEAREST", dst)

dst = cv.resize(src, (w, h), fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR)
# 图像输出格式与原图一样 图像缩放系数为0.25 使用像素区域关系进行重采样。
cv.imshow("INTER_LINEAR", dst)

dst = cv.resize(src, (w*2, h*2), interpolation=cv.INTER_CUBIC)
# 图像输出格式为原图的长与宽的0.2  插入方式为4x4像素邻域的双三次插值
cv.imshow("INTER_CUBIC", dst)

dst = cv.resize(src, (w*5, h*5), fx=5, fy=5, interpolation=cv.INTER_LANCZOS4)
# 图像输出大小设为原图1/2 图像缩放系数为原图1/2  使用8x8像素邻域的Lanczos插值
cv.imshow("INTER_LANCZOS4", dst)

# cv.warpAffine()
# wapAffine()可以实现图像的仿射变换

cv.waitKey(0)
cv.destroyAllWindows()

