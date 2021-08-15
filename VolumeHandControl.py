import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import osascript

##################################################
wCam, hCam = 640, 488
##################################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
vol = 0
volBar = 400

detector = htm.handDetector(detectionCon=0.7)

def setVolume(vol):
    osascript.osascript("set volume output volume {}".format(vol))

# code, out, err = osascript.run("output volume of (get volume settings)")
# print(out)

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2) // 2, (y1+y2) // 2

        cv2.circle(img, (x1, y1), 12, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 12, (255, 0, 255), cv2.FILLED)

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 12, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)

        # Hand range 20 - 200
        # Volume Range 0 - 10
        vol = np.interp(length, [20,200], [0, 100])
        volBar = np.interp(length, [20,200], [400, 150])
        print(vol)
        setVolume(vol)

        if length < 50:
            cv2.circle(img, (cx, cy), 12, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol)}%', (40,450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    if k == 27:
        exit()