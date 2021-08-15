import cv2
import time
import numpy as np

#####################################
ECS_KEY_CODE = 27
thres = 0.45 # Threshold to detect object
nms_threshold = 0.2
#####################################

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 150)

pTime = 0

classNames = []
with open("ssd_mobilenet_model/coco.names", "rt") as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = "ssd_mobilenet_model/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightPath = "ssd_mobilenet_model/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    # img = cv2.imread("Header-Files/lena.png")

    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))
    
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]

        cv2.rectangle(img, (x,y), (x+w, h+y), color=(0,255,0), thickness=2)
        cv2.putText(img, f'{classNames[classIds[i][0]-1].upper()}', (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    # if len(classIds) > 0:
    #     for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    #         cv2.rectangle(img, box, color=(0,255,0), thickness=2)
    #         cv2.putText(img, f'{classNames[classId-1].upper()} {int(confidence*100)}%', (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)

    cv2.imshow("Object Detector", img)
    k = cv2.waitKey(1)
    if k == ECS_KEY_CODE:
        exit()