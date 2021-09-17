import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import imutils
from solver import *

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# classes = [7, 3, 6, 1, 0, 4, 8, 2, 5, 9]
model = load_model('model-OCR.h5')
print(model.summary())
input_size = 48

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

def split_boxes(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size))/255.0
            boxes.append(box)
    return boxes

def displayNumbers(img, numbers, color=(0, 255, 0)):
    W = int(img.shape[1]/9)
    H = int(img.shape[0]/9)
    for i in range (9):
        for j in range (9):
            if numbers[(j*9)+i] !=0:
                cv2.putText(img, str(numbers[(j*9)+i]), (i*W+int(W/2)-10, int((j+0.8)*H)), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
    return img


board = find_board(cv2.imread('suduku3.jpg'))
gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
print(gray.shape)
rois = split_boxes(gray)
print(len(rois))
print(classes)
print(len(classes))

rois = np.array(rois).reshape(-1, input_size, input_size, 1)
print(rois.shape)
prediction = model.predict(rois)
print(prediction)


predicted_numbers = []
for i in prediction:
    
    index = (np.argmax(i))
    predicted_number = classes[index]
    predicted_numbers.append(predicted_number)

print(predicted_numbers)
binArr = np.where(np.array(predicted_numbers)>0, 0, 1)
print(binArr)
board = np.array(predicted_numbers).astype('uint8').reshape(9, 9)
print(solve(board))
print_board(board)
cv2.waitKey(0)

cv2.destroyAllWindows()