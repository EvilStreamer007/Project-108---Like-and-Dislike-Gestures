import cv2
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

fingerTips =[8, 12, 16, 20]
thumbTip= 4

while True:
    ret,img = cap.read()
    img = cv2.flip(img, 1)
    h,w,c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list=[]
            for id ,lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            finger_fold_status =[]
            for tip in fingerTips:
                x,y = int(lm_list[tip].x*w), int(lm_list[tip].y*h)
                cv2.circle(img, (x,y), 15, (255, 0, 0), cv2.FILLED)

                if lm_list[tip].x < lm_list[tip - 3].x:
                    cv2.circle(img, (x,y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            print(finger_fold_status)

            if all(finger_fold_status):   
                if lm_list[thumbTip].y < lm_list[thumbTip-1].y < lm_list[thumbTip-2].y:
                    print("Like")  
                    cv2.putText(img ,"Like", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

                if lm_list[thumbTip].y > lm_list[thumbTip-1].y > lm_list[thumbTip-2].y:
                    print("Dislike")   
                    cv2.putText(img ,"Dislike", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            mpDraw.draw_landmarks(img, hand_landmark,mpHands.HAND_CONNECTIONS, mpDraw.DrawingSpec((0,0,255),2,2),
                                    mpDraw.DrawingSpec((0,255,0),4,2))
    

    cv2.imshow("Hand Tracking", img)
    cv2.waitKey(1)