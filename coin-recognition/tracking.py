import cv2 # pip install opencv-python"
import numpy as np
from pathlib import Path

# Esc キー
ESC_KEY = 0x1b
# s キー
S_KEY = 0x73
# r キー
R_KEY = 0x72

BASE_DIR = Path(__file__).resolve().parent

CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

INTERVAL = 100
interval = 0

points = []

def mouseEvents(event, x, y, flags, param):
    try:
        if event == cv2.EVENT_LBUTTONDOWN:
            add_feature(x, y)
    except Exception as e:
        print(e)

def add_feature(x, y):
    global points
    points.append([[x, y]])  
   

while True:
    img = cv2.imread(BASE_DIR / "infinicam_coin_toss_meetingroom_10yen_1000fps" / "800.bmp")
    cv2.imshow('test', img)
    cv2.setMouseCallback('test', mouseEvents)
    key1 = cv2.waitKey(interval)
    if key1 == ESC_KEY: #画面を閉じる
        break
    elif key1 == R_KEY: #再生 
        interval = INTERVAL
    
    for i in range(800, 1400):
        number = str(i) + '.bmp' 
        next_number = str(i + 1) + '.bmp' 
        img = cv2.imread(BASE_DIR / "infinicam_coin_toss_meetingroom_10yen_1000fps" / number)
        next_img = cv2.imread(BASE_DIR / "infinicam_coin_toss_meetingroom_10yen_1000fps" / next_number)

        gray_i = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_ni = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

        p0 = np.array(points, dtype=np.float32)

        cv2.cornerSubPix(gray_ni, p0, (10, 10), (-1, -1), CRITERIA)

        p1, st, err = cv2.calcOpticalFlowPyrLK(gray_i, gray_ni, p0, None, winSize = (10, 10), maxLevel = 3, criteria = CRITERIA, flags = 0)

        x, y = p1[0].ravel()  
        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
        
        cv2.imshow('test', img)
        key2 = cv2.waitKey(interval)
        if key2 == ESC_KEY: #最初に戻る
            interval = 0
            break
        elif key2 == S_KEY: #一時停止
            interval = 0
        elif key2 == R_KEY: #再生
            interval = INTERVAL

cv2.destroyAllWindows()