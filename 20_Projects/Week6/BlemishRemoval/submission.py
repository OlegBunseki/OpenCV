# Enter your code here
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


def calc_lap(img, center, r):
    
    sel = img[center[0]-r:center[0]+r, center[1]-r:center[1]+r] 
    sel_gray = cv2.cvtColor(sel, cv2.COLOR_BGR2GRAY) 
    lap = cv2.Laplacian(sel_gray, cv2.CV_32F)
    
    return sel_gray, lap


def find_best_center(img, center, r):

    
    gr, lap = calc_lap(img=img, center=center, r=r)
    min_var = np.var(lap)
    center_min_var = center
    
    moves = [(-1, 1), (-1, 0), (-1, -1), (0, 1), (0, -1), (1, -1), (1, 0), (1, 1)]
    
    for m in moves:
        
        new_center = (center[0]+m[0]*r*2, center[1]+m[1]*r*2)
        
        gr, lap = calc_lap(img=img, center=new_center, r=15)
        
        if np.var(lap) < min_var: 
            min_var = np.var(lap)
            center_min_var = new_center
            
    return center_min_var


def onMouse(action, x, y, flags, img):
    
    if action == cv2.EVENT_LBUTTONDOWN:
        
        center = (y, x)
        r = 15
        
        new_center_point = find_best_center(img, center, r)
        
        colorpatch = img[new_center_point[0]-r:new_center_point[0]+r, new_center_point[1]-r:new_center_point[1]+r]
        src_mask = np.ones_like(colorpatch[:,:,0]) * 255        
        cv2.seamlessClone(colorpatch, img, src_mask, (center[1], center[0]), cv2.NORMAL_CLONE, blend=img)
        
        
img = cv2.imread("blemish.png", 1)
cv2.namedWindow("Window")
cv2.setMouseCallback("Window", onMouse, img)

k = 0
while k != 27:
    cv2.imshow("Window", img)        
    k = cv2.waitKey(20)
    
cv2.destroyAllWindows()