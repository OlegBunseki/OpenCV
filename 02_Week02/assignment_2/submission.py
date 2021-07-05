import cv2
import os


def path_append_i(path, i):
    return '.'.join([path.split('.')[0]+'_'+str(i), path.split('.')[1]])
    

def crop_face(action, x, y, flags, userdata):
    
    global point1, point2, i, image_path
    
    th = 2
    
    if action == cv2.EVENT_LBUTTONDOWN:
        point1 = [(x, y)]
    
    if action == cv2.EVENT_LBUTTONUP:
        point2 = [(x, y)]
        
        cv2.rectangle(image, point1[0], point2[0], color=(255, 0, 255), thickness=th)

        start = ( min(point1[0][0], point2[0][0]), min(point1[0][1], point2[0][1]) )
        end = ( max(point1[0][0], point2[0][0]), max(point1[0][1], point2[0][1]) )
        
        # Adjust the start and ending point, so the rectangle is not seen in the picture
        face = image[start[1]+th:end[1]-th, start[0]+th:end[0]-th]
        
        save_path = image_path
   
        while os.path.exists(save_path):
            save_path = path_append_i(image_path, i)
            i += 1
        
        cv2.imwrite(save_path, face)
            
        cv2.imshow('Window', image)
        

if __name__ == '__main__': 
    image = cv2.imread('Image_Boy.jpg', 1)
    dummy = image.copy()

    point1, point2 = [], []
    i = 1
    path = os.getcwd()
    image_name = 'face.jpg'
    image_path = os.path.join(path, image_name)

    cv2.namedWindow('Window')
    cv2.setMouseCallback('Window', crop_face)
    k = 0

    while k != 27:
        
        cv2.imshow('Window', image)
        
        # Read keyboard input
        k = cv2.waitKey(1) & 0xFF
        
        # break the loop
        if k == 27:
            break 
            
        # Clean the picture
        if k==99:
            source = dummy.copy()
            
    cv2.destroyAllWindows()