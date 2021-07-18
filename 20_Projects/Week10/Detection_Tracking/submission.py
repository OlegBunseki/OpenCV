# Enter your code here
import cv2
import numpy as np
import argparse
import time


def drawTrack(box):

    x, y, w, h = box

    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 3)



# Draw the predicted bounding box
def drawPred(classId, conf, box):

    x, y, w, h = box

    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, labelSize[1])
    
    cv2.rectangle(frame, (x, y - round(1.5*labelSize[1])), (x + round(1.5*labelSize[0]), y + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)


def process(frame):

    outs = run_detection(frame)

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            if detection[4] > args.objectnessThreshold :
                scores = detection[5:]
                classId = np.argmax(scores)

                if classId != 32:
                    continue

                confidence = scores[classId]

                if confidence > args.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    w = int(detection[2] * frameWidth)
                    h = int(detection[3] * frameHeight)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

    indices = [i[0] for i in cv2.dnn.NMSBoxes(boxes, confidences, args.confThreshold, args.nmsThreshold)]

    for i in indices:
        drawPred(classIds[i], confidences[i], boxes[i])

    if len(boxes)>0:
        return True, boxes[0]
    else:
        return False, []


def run_detection(frame):

    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    return outs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--video_path', type=str, default='soccer-ball.mp4', help='Video Path')
    parser.add_argument('-objt', '--objectnessThreshold', type=float, default='0.5', help='Objectness Threshold')    
    parser.add_argument('-cont', '--confThreshold', type=float, default='0.5', help='Confidence Threshold')
    parser.add_argument('-nms', '--nmsThreshold', type=float, default='0.4', help='NMS Threshold')
    parser.add_argument('-tt', '--TrackerType', type=str, default='KCF', help='Tracker Type')
    parser.add_argument('-dp', '--dataPath', type=str, default='./data/', help='Data Path')
    parser.add_argument('-mp', '--modelPath', type=str, default='./weights/', help='Model Path')


    args = parser.parse_args()

    DATA_PATH = args.dataPath
    MODEL_PATH = args.modelPath

    if args.TrackerType == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif args.TrackerType == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif args.TrackerType == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = MODEL_PATH + "yolov3.cfg"
    modelWeights = MODEL_PATH + "yolov3.weights"

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

    VIDEO_PATH = DATA_PATH + args.video_path

    # Parameters
    inpWidth = 416            # Width of network's input image
    inpHeight = 416           # Height of network's input image

    # Load names of classes
    classesFile = MODEL_PATH + "coco.names"

    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    cap = cv2.VideoCapture(VIDEO_PATH)

    FPS = int(cap.get(cv2.CAP_PROP_FPS))  
  
    
    frame_count = 0
    track_status = False

    while cap.isOpened():

        ret, f = cap.read()

        frame = f.copy()

        if ret == False:
            break

        # Initialize
        if frame_count == 0:

            ball_found, ball_bbox = process(frame)
            if ball_found:
                init = tracker.init(frame, ball_bbox)
                track_status = True
                
        if (frame_count > 0) and (ball_found is True):
            track_status, ball_bbox = tracker.update(frame)

        # Tracker
        if init is None:
            pass
        
        elif track_status:
            cv2.putText(frame, f'Tracking is running: {track_status}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            drawTrack(ball_bbox)

        else:
            cv2.putText(frame, f'Tracking is running: {track_status}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            ball_found, ball_bbox = process(frame)
            if ball_found:
                tracker = cv2.TrackerKCF_create()
                init = tracker.init(frame, ball_bbox)
                track_status = True
                print('3', frame_count, track_status)

            cv2.putText(frame, f'Ball could be detected: {ball_found}', (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow('Video', frame)

        # Make the visualisation a little slower
        # time.sleep(1)

        k = cv2.waitKey(1)

        if k == 27:
            break

        init = False
        frame_count += 1
