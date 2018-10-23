"""
Chương trình thử nghiệm phương pháp SIFT vào việc phát hiện đối tượng trong ảnh
Các bước thực hiện:
- Đọc 2 ảnh.
- Thực hiện SIFT để có được keypoint và descriptor
- Thực hiện FLANN matcher để tìm điểm khớp của 2 ảnh.
- 

change log:
    26/7/2008: triển khai thử nghiệm SIFT, dùng Flann matcher theo thông số Knn.
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt


MIN_MATCH_COUNT = 10 #số điểm match tối thiểu để xác nhận khớp
img1 = cv2.imread('./data/sach1.jpg',0)          # ảnh huấn luyện
img2 = cv2.imread('./data/sach2.jpg',0)            # ảnh cần tìm
# khởi tạo SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# tìm keypoints và descriptors cho 2 ảnh dùng SIFT
#SURF cũng thực hiện tương tự
kp1, des1 = sift.detectAndCompute(img1,None) 
kp2, des2 = sift.detectAndCompute(img2,None)
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



#chỉ nhận dạng khi đủ số lượng match > MIN_MATCH_COUNT (=10)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) #thực hiện tìm Homography trên 2 ảnh.
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
#đưa ra cảnh báo về độ matching yếu
else:
    print("Tim khong du so diem khop toi thieu - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # màu để vẽ đường match RGB(0,255,0) 
                   singlePointColor = None,
                   matchesMask = matchesMask, # Chỉ vẽ những đường trong.
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)# tạo ảnh có cách đường matching.
plt.figure('Ket qua SIFT tren 2 anh')
plt.imshow(img3, 'gray'),plt.show()




