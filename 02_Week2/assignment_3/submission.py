import cv2


# Callback functions
def scaleImage(*args):

    global scaleType
    global scaleFactor
    
    # Check if Up- or Downscaling
    if scaleType == 0:
        factor = 1
    else:
        factor = -1
    
    
    # Get the scale factor from the trackbar 
    scaleFactor = 1 + factor*args[0]/100.0
    # print('scaleImage scaleFactor', scaleFactor)
    
    # Perform check if scaleFactor is zero
    if scaleFactor == 0:
        scaleFactor = 1
    
    # Resize the image
    scaledImage = cv2.resize(im, None, fx=scaleFactor, fy = scaleFactor, interpolation = cv2.INTER_LINEAR)
    
    cv2.imshow(windowName, scaledImage)


def scaleTypeImage(*args):
    
    global scaleType
    global scaleFactor
    
    scaleType = args[0]
    # print('scaleTypeImage scaleType', scaleType)
    
    # Check if Up- or Downscaling
    if scaleType == 0:
        factor = 1
    else:
        factor = -1
    
    # Get the scale factor from the trackbar 
    scaleFactor = 1 + factor*args[0]/100.0
    # print('scaleTypeImage scaleFactor', scaleFactor)
    
    if scaleFactor == 0:
        scaleFactor = 1
        
    scaledImage = cv2.resize(im, None, fx=scaleFactor, fy = scaleFactor, interpolation = cv2.INTER_LINEAR)
    
    cv2.imshow(windowName, scaledImage)
    

if __name__ == '__main__':   
 
    maxScaleUp = 100
    scaleFactor = 1
    scaleType = 0
    maxType = 1

    windowName = "Resize Image"
    trackbarValue = "Scale"
    trackbarType = "Type: \n 0: Scale Up \n 1: Scale Down"


    # load an image
    im = cv2.imread("truth.png")

    # Create a window to display results
    cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar(trackbarValue, windowName, scaleFactor, maxScaleUp, scaleImage)
    cv2.createTrackbar(trackbarType, windowName, scaleType, maxType, scaleTypeImage)

    cv2.imshow(windowName, im)
    c = cv2.waitKey(0)

    cv2.destroyAllWindows()