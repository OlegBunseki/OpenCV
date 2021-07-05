import cv2
import os

zero = cv2.imread('./data/images/boy.jpg')
cv2.imshow('Zero', zero)
cv2.waitKey(0)
cv2.destroyAllWindows()