import numpy as np
import cv2
import time
import HandTrackingModule as htm
import math
import osascript

wCam, hCam = 512, 420

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

minVol =0
maxVol = 100

# result = osascript.osascript('get volume settings')
# print(result)

volume = 0
volBar = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    # img_flip = cv2.flip(img, 1)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList)!=0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        # print(length)

        # Hand rage 50 - 300
        # Volume Range 0 - 100

        volume = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        # print(volume)

        # target_volume = 0

        vol = "set volume output volume " + str(volume)
        osascript.osascript(vol)

        if length<50:
            cv2.circle(img, (cx, cy), 15, (0,255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (75, 375), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (75, 375), (0, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'Fps:{int(fps)}', (50, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)


