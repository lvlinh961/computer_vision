import cv2
import numpy as np
import time
import PoseEstimateModule as pem

cap = cv2.VideoCapture("gym.mp4")

detector = pem.poseDetector()
count = 0
dir = 0

pTime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))

    # img = cv2.imread("gym_exercise.png")
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Right Arm
        angle = detector.findAngle(img, 12, 14, 16)
        # Left Arm
        # angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (200, 360), (0, 100))
        bar = np.interp(angle, (200, 360), (650, 100))

        # Check for the dumbbell curls
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir=1

        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        print(int(per), count)

        # Draw bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 2)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (1120, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        # Draw Curl Count
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{count}', (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    cv2.imshow("AI Trainer", img)
    k = cv2.waitKey(1)
    if k == 27:
        exit()
