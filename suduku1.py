import cv2
import numpy as np
import imutils
import time
import uuid

def get_perspective(img, location, height = 900, width = 900):
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result

def find_board(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours  = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break
    result = get_perspective(img, location)
    return result

input_size = 128
def cropped_roi(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size))
            boxes.append(box)
    return boxes

board = find_board(cv2.imread('suduku4.jpg'))
board = cv2.resize(board, (900, 900))
gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
rois = cropped_roi(gray)
print(len(rois))


# rois = np.array(rois).reshape(-1, input_size, input_size, 1)
# print(rois.shape)

for i in rois:
    
    cv2.imshow("View", i)
    cv2.waitKey(1)
    print(i.shape)
    f_name = 'collected/'+str(uuid.uuid4())+'.jpg'
    cv2.imwrite(f_name, i)
    # print(i.shape)
    # i = np.expand_dims(i, axis = 0)
    # print(i.shape)

    # time.sleep(1)


cv2.waitKey(0)

cv2.destroyAllWindows()