import cv2 # pip install opencv-python"
import time
import numpy as np
import openvino as ov
from pathlib import Path

cap = cv2.VideoCapture(0)

fontType = cv2.FONT_HERSHEY_DUPLEX

while True:
# ウェブカメラの画像取得
 ret, img = cap.read()

 # グレースケール画像の生成
 image_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

 # Cannyでエッジ検出処理
 canny_gray = cv2.Canny(image_gray,100,200)
 cimg = canny_gray

 j = 0

    # hough関数
 circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 20, param1 = 120, param2 = 15, minRadius = 10,maxRadius = 30)
        # param1 : canny()エッジ検出器に渡される2つの閾値のうち、大きいほうの閾値0
        # param2 : 円の中心を検出する際の投票数の閾値。小さくなるほど、誤検出が起こる可能性がある。
        # minRadius : 検出する円の最小値
        # maxRadius : 検出する円の最大値

    #検出された際に動くようにする。
 if circles is not None and len(circles) > 0:

        #型をfloat32からunit16に変更。
    circles = np.uint16(np.around(circles))
        
    for i in circles[0,:]:
            # 外側の円を描く
        cv2.circle(img,(i[0], i[1]), i[2], (0, 191, 255), 2)
            # 中心の円を描く
        cv2.circle(img,(i[0], i[1]), 2, (255, 255, 0), 2)
            # 円の数を数える
        j = j + 1

    #円の合計数を表示
 cv2.putText(img,'Total :'+str(j), (30,30), fontType, 1, (0, 0, 0), 1, cv2.LINE_AA)

# 画像表示
 cv2.imshow('test', img)
 cv2.waitKey(1) #待機時間、ミリ秒指定、0の場合はボタンが押されるまで待機

 key = cv2.waitKey(1) & 0xFF
 if key == 27:                                # Esc:終了
    break



cap.release()
cv2.destroyAllWindows()