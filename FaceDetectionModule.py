import cv2
import mediapipe as mp
import time

class faceDetection():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFace(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                h, w, c = img.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                        int(bboxC.width * w), int(bboxC.height * h)
                bboxs.append([id, bbox, detection.score])
                
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', 
                        (bbox[0], bbox[1] - 20), 
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2
                    )
        return img, bboxs

    def fancyDraw(self, img, box, l=30, t=5, rt=1):
        x, y, w, h  = box
        x1, y1 = x  + w, y + h

        cv2.rectangle(img, box, (255, 0, 255), rt)

        # Top left x,y
        cv2.line(img, (x,y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x,y), (x, y + l), (255, 0, 255), t)
        # Top right x1,y
        cv2.line(img, (x1,y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1,y), (x1, y + l), (255, 0, 255), t)
        # Bottom left x,y1
        cv2.line(img, (x,y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x,y1), (x, y1 - l), (255, 0, 255), t)
        # Bttom right x1,y1
        cv2.line(img, (x1,y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1,y1), (x1, y1 - l), (255, 0, 255), t)

        return img


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0

    while True:
        success, img = cap.read()
        detection = faceDetection()
        img, bboxs = detection.findFace(img, True)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()