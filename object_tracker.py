#%%
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#%%
# find SIFT features in images
# apply the ratio test to find the best matches.
MIN_MATCH_COUNT = 10
# a condition that at least 10 matches are to be there to find the object,
# otherwise simply show a message saying not enough matches are present.
img1 = cv.imread("photo_3_query.jpg", cv.IMREAD_GRAYSCALE) # queryImage
vid = cv.VideoCapture('video_3_train.mp4',cv.IMREAD_GRAYSCALE) # trainVideo
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict()
flann = cv.FlannBasedMatcher(index_params, search_params)

while vid.isOpened():
    # reading the frame  
    ret, frame = vid.read() 
    if not ret:
        print("Can't recieve frame. Exiting.....")
        break
    
    # converting the frame into grayscale 
    grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    
    # find the keypoints and descriptors with SIFT 
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None) 
    
    # finding nearest match with KNN algorithm 
    matches= flann.knnMatch(des1, desc_grayframe, k=2) 
    
    # initialize list to keep track of only good points 
    good=[] 
    
    for m, n in matches: 
        #append the points according 
        #to distance of descriptors 
        if(m.distance < 0.7*n.distance): 
            good.append(m) 

    # maintaining list of index of descriptors 
    # in query descriptors 
    query_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) 
    
    # maintaining list of index of descriptors 
    # in train descriptors 
    train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) 
    
    # finding  perspective transformation 
    # between two planes 
    matrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0) 
    
    # ravel function returns  
    # contiguous flattened array 
    matches_mask = mask.ravel().tolist()


    # initializing height and width of the image 
    h, w = img1.shape 
    
    # saving all points in pts 
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2) 

    if matrix is not None:
        # applying perspective algorithm 
        dst = cv.perspectiveTransform(pts, matrix) 

        # using drawing function for the frame 
        homography = cv.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3) 
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matches_mask, # draw only inliers
                    flags = 2)
        img3 = cv.drawMatches(img1,kp1,homography,kp_grayframe,good,None,**draw_params)
        cv.imshow("Homography", img3) 
        if cv.waitKey(1) == ord("q"):
            break
vid.release()
cv.destroyAllWindows()