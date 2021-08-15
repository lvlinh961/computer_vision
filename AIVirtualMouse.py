import cv2
import numpy as  np
import time
import HandTrackingModule as htm
import autopy

##############################
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 10
##############################

pTime = 0
plocX, plocY = 0,0
clocX, clocY = 0,0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector()

wScr, hScr = autopy.screen.size()
print(wScr, hScr)

while True:
    success, img = cap.read()

    # 1. Find hand Landmarks
    img = detector.findHands(img)
    # img = cv2.flip(img, 1)
    lmList = detector.findPosition(img, draw=False)

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        cv2.rectangle(img, (frameR,frameR), (wCam-frameR, hCam-frameR), (255,0,255), 2)

        # 4. Only Index Finger: Moving Mode
        if fingers[1] and fingers[2] == False:
            # 5. Convert Coordicates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocX) / smoothening

            # 7. Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 8. Both Index and Middle fingers are up: Clicking Mode
        if fingers[1] and fingers[2]:
            # 9. Find dstance between fingers
            length, img, lineInfo = detector.findDistance(8,12, img)
            if length < 40:
                cv2.circle(img, (lineInfo[4],lineInfo[5]), 15, (0,255,0), cv2.FILLED)
                # 10. Click mouse if distance short
                autopy.mouse.click()

        
        

    # 11. Frame Rate
    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display

    cv2.imshow("Vitual Mouse", img)
    k = cv2.waitKey(1)
    if k == 27:
        exit()