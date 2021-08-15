import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 488
blueColor = (255, 0, 0)

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        print("Finger count: {}".format(fingers.count(1)))
        cv2.putText(img, str(fingers.count(1)), (50, 70), cv2.FONT_HERSHEY_PLAIN, 3, blueColor, 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {fps}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, blueColor, 3)

    cv2.imshow("Finger Counter", img)
    k = cv2.waitKey(1)

    if k == 27:
        exit()