# Enter your code here
import cv2
import numpy as np
import argparse

# https://learnopencv.com/applications-of-foreground-background-separation-with-semantic-segmentation/
# https://www.compuphase.com/cmetric.htm
# https://smirnov-am.github.io/chromakeying/


def process_patches(image, background, dev, patches, g_kernsel_size):
    
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    if len(patches) == 0:
        lower_hue, upper_hue = (54, 54)
        lower_sat, upper_sat = (237, 237)
        lower_val, upper_val = (255, 255)
        
    else:
        lower_hue, upper_hue = ( np.min(patches[:,0]), np.max(patches[:,0]) ) 
        lower_sat, upper_sat = ( np.min(patches[:,1]), np.max(patches[:,1]) )
        lower_val, upper_val = ( np.min(patches[:,2]), np.max(patches[:,2]) )
    
    # lower_bound = np.array([ lower_hue-lower_hue*dev[0]/100, lower_sat-lower_sat*dev[1]/100, lower_val-lower_val*dev[2]/100 ])
    # upper_bound = np.array([ upper_hue+upper_hue*dev[0]/100, upper_sat+upper_sat*dev[1]/100, upper_val+upper_val*dev[2]/100 ])

    # print(dev)
    lower_bound = np.array([ lower_hue-lower_hue*dev/100, lower_sat-lower_sat*dev/100, lower_val-lower_val*dev/100 ])
    upper_bound = np.array([ upper_hue+upper_hue*dev/100, upper_sat+upper_sat*dev/100, upper_val+upper_val*dev/100 ])
    #     
    # print(lower_bound, upper_bound)
    
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)    
    # mask1d = np.uint8(mask/255)


    # blur mask
    if (g_kernsel_size == 1) | (g_kernsel_size % 2 != 0):
        mask_blur = cv2.GaussianBlur(mask, (g_kernsel_size, g_kernsel_size), 0, 0)
    
    elif g_kernsel_size % 2 == 0:
        g_kernsel_size += 1
        mask_blur = cv2.GaussianBlur(mask, (g_kernsel_size, g_kernsel_size), 0, 0)

    mask1d = np.uint8(mask_blur/255)


    ##############################################################################################

    mask3d = cv2.merge((mask1d, mask1d, mask1d))
    
    
    visible_objects_hsv = cv2.multiply(np.array(image_hsv, dtype=float), np.array(1-mask3d, dtype=float))
    visible_objects_bgr = cv2.cvtColor(np.uint8(visible_objects_hsv), cv2.COLOR_HSV2BGR)
    background_masked =  np.uint8(cv2.multiply(np.array(background, dtype=float), np.array(mask3d, dtype=float)))

    # visible_objects_hsv = cv2.multiply(image_hsv, 1-mask3d)
    # visible_objects_bgr = cv2.cvtColor(visible_objects_hsv, cv2.COLOR_HSV2BGR)
    # background_masked = cv2.multiply(background, mask3d)
    
    removed_green_screen = cv2.add(background_masked, visible_objects_bgr)
    
    return removed_green_screen


# def calc_blur(image, i):

#     if (i == 1) | (i % 2 != 0):
#         blurred_image = cv2.GaussianBlur(image, (i, i), 0, 0)
    
#     elif i % 2 == 0:
#         i += 1
#         blurred_image = cv2.GaussianBlur(image, (i, i), 0, 0)

#     return blurred_image


def get_color_patch(action, x, y, flags, userdata):

    global patches
    
    if action==cv2.EVENT_LBUTTONDOWN:
        
        # center=[(x,y)]
        
        print(x, y)
        
        patch = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[y,x,:]
        print('patch', patch)
        
        patches = np.append(patches, patch).reshape(-1, 3)
        print('patches', patches)
        
        #print('RGB:', image[y,x,:])
        #print('HSV', cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[y,x,:])


def mod_green_dev(*args):
    global green_dev
    global patches    
    global g_kernsel_size

    green_dev = args[0]
    
    processed_image = process_patches(image, background_scaled, dev=green_dev, patches=patches, g_kernsel_size=g_kernsel_size) 
    
    cv2.imshow(video_window_name, processed_image)

# def mod_green_hue(*args):
#     global hue_dev
#     global saturation_dev
#     global value_dev
#     global patches    
#     global g_kernsel_size

#     hue_dev = args[0]
    
#     processed_image = process_patches(image, background_scaled, dev=(hue_dev, saturation_dev, value_dev), patches=patches, g_kernsel_size=g_kernsel_size) 
    
#     cv2.imshow(video_window_name, processed_image)
    
# def mod_green_saturation(*args):
#     global hue_dev
#     global saturation_dev
#     global value_dev
#     global patches    
#     global g_kernsel_size

#     saturation_dev = args[0]

#     processed_image = process_patches(image, background_scaled, dev=(hue_dev, saturation_dev, value_dev), patches=patches, g_kernsel_size=g_kernsel_size) 
    
#     cv2.imshow(video_window_name, processed_image)

    
# def mod_green_value(*args):
#     global hue_dev
#     global saturation_dev
#     global value_dev
#     global patches    
#     global g_kernsel_size

#     value_dev = args[0]

#     processed_image = process_patches(image, background_scaled, dev=(hue_dev, saturation_dev, value_dev), patches=patches, g_kernsel_size=g_kernsel_size) 
    
#     cv2.imshow(video_window_name, processed_image)


def blur(*args):
    # global hue_dev
    # global saturation_dev
    # global value_dev
    global green_dev
    global patches    
    global g_kernsel_size

    green_dev = args[0]
    
#    processed_image = process_patches(image, background_scaled, dev=(hue_dev, saturation_dev, value_dev), patches=patches, g_kernsel_size=g_kernsel_size) 
    processed_image = process_patches(image, background_scaled, dev=green_dev, patches=patches, g_kernsel_size=g_kernsel_size)     
    cv2.imshow(video_window_name, processed_image)


def save_video(cap, video_local_path):
    
    codec = cv2.VideoWriter_fourcc('M','J','P','G')
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(video_local_path, codec, fps, (width, height))
    
    return out


# python submission.py -v "./data/greenscreen-demo.mp4"

if __name__ == '__main__':

    ################################################################################################################
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', type=str, default='greenscreen-asteroid.mp4')
    args = parser.parse_args()

    ################################################################################################################


    cap = cv2.VideoCapture(args.video)
    out = save_video(cap, 'test.mov')

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    setting_window_name = 'Settings'
    video_window_name = 'Video'

    frame_count = 1

    background = cv2.imread('./data/sky.jpg', cv2.IMREAD_COLOR)
    background_scaled = cv2.resize(background, (width, height))

    cv2.namedWindow(setting_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(setting_window_name, 900, 600)
    cv2.namedWindow(video_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(video_window_name, 900, 600)

    cv2.setMouseCallback(setting_window_name, get_color_patch)

    green_dev = 0
    # hue_dev, saturation_dev, value_dev = (0, 0, 0)
    g_kernsel_size = 1
    patches = np.array([])

    # cv2.createTrackbar('hue', setting_window_name, 0, 100, mod_green_hue)
    # cv2.createTrackbar('saturation', setting_window_name, 0, 100, mod_green_saturation)
    # cv2.createTrackbar('value', setting_window_name, 0, 100, mod_green_value)
    
    cv2.createTrackbar('green deviation', setting_window_name, 0, 100, mod_green_dev)    
    cv2.createTrackbar('blur', setting_window_name, 1, 35, blur)


    while cap.isOpened():  
        ret, image = cap.read()  
        
        if ret:   
            
            if frame_count == 1:
                first_image = image.copy()
                
            # blurred = calc_blur(image, i)
            # processed_image = process_patches(image, background_scaled, dev=(hue_dev, saturation_dev, value_dev), patches=patches, g_kernsel_size=g_kernsel_size ) 
            
            processed_image = process_patches(image, background_scaled, dev=green_dev, patches=patches, g_kernsel_size=g_kernsel_size )             
            cv2.imshow(video_window_name, processed_image)
            
            cv2.imshow(setting_window_name, first_image)       
            
            out.write(processed_image)
            frame_count += 1
            
            k = cv2.waitKey(int(1000/fps))

            if k == 27:
                break        
        else:
            break

    out.release()      
    cap.release()
    cv2.destroyAllWindows()