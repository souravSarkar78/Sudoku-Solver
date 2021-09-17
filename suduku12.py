import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from solver import *

# classes = np.arange(0, 10)
classes = [7, 3, 6, 1, 0, 4, 8, 2, 5, 9]
model = load_model('model.h5')
print(model.summary())
input_size = 48

def cropped_roi(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size))/255.0
            boxes.append(box)
    return boxes

board = cv2.imread('suduku1.png', 0)
board = cv2.resize(board, (900, 900))
rois = cropped_roi(board)
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
board = np.array(predicted_numbers).astype('uint8').reshape(9, 9)
print(solve(board))
print_board(board)
cv2.waitKey(0)

cv2.destroyAllWindows()