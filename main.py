import cv2 as cv
import mediapipe as mp
import time
import math


cap = cv.VideoCapture(0)
mphands=mp.solutions.hands
hands=mphands.Hands()
mpDraw= mp.solutions.drawing_utils


pTime=0
cTime=0

while True:
    success,img = cap.read()

    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results= hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handlmks in results.multi_hand_landmarks:
            temp1= [0,0]
            temp2= [0,40]
            for id,lm in enumerate(handlmks.landmark):
                h,w,c = img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                if id ==4:
                    temp1= [cx,cy]
                if id == 8:
                    temp2= [cx,cy]

                dist = math.dist(temp1,temp2)
                print(dist)
                if dist<30:
                    print('noice')


            mpDraw.draw_landmarks(img,handlmks,mphands.HAND_CONNECTIONS)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_ITALIC,3,(0,0,255),3)


    cv.imshow('image',img)
    cv.waitKey(1)