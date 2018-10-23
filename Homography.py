"""
Chương trình thử nghiệm dùng Homography để đồng chỉnh ảnh theo một ảnh mẩu. Việc đồng
chỉnh được thực hiện dựa trên các điểm mốc keypoint.
Keypoint sử dụng SIFT để tìm

change log:
    23/10/2018: triển khai thử nghiệm SIFT, dùng Homography để đồng chỉnh ảnh
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt


MIN_MATCH_COUNT = 10 #số điểm match tối thiểu để xác nhận khớp
img1 = cv2.imread('./data/sach1.jpg')          # ảnh huấn luyện
img2 = cv2.imread('./data/sach2.jpg')            # ảnh cần tìm

#chuyển qua ảnh xám
im1gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
im2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# khởi tạo SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# tìm keypoints và descriptors cho 2 ảnh gray dùng SIFT
kp1, des1 = sift.detectAndCompute(im1gray,None) 
kp2, des2 = sift.detectAndCompute(im2gray,None)


# Thực hiện matcher trên keypoint(kp) và descriptor(des) mà SIFT cho ra.
# Matcher sử dụng: Flann matcher. Phương thức matcher dùng Knn 
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50) #xác định số lần đệ qui duyệt qua cây.

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Giữ lại những đặc trưng tốt theo tỉ lệ match của  Lowe trong paper.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


#vẽ những đặc trưng khớp của 2 ảnh.
imMatches=cv2.drawMatches(im1gray,kp1,im2gray,kp2,good,None)
plt.imshow(imMatches)
plt.show()



