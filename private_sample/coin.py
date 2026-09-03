import cv2
import numpy as np

# カメラを開く
cap = cv2.VideoCapture(1)

#フォントの指定
fontType = cv2.FONT_HERSHEY_DUPLEX

while True:

    # 画像をキャプチャする
    ret, frame = cap.read() #カメラ動画

    # グレースケール画像の生成
    image_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #==============================================
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
            cv2.circle(frame,(i[0], i[1]), i[2], (0, 191, 255), 2)
            # 中心の円を描く
            cv2.circle(frame,(i[0], i[1]), 2, (255, 255, 0), 2)
            # 円の数を数える
            j = j + 1

    #円の合計数を表示
    cv2.putText(frame,'Total :'+str(j), (30,30), fontType, 1, (0, 0, 0), 1, cv2.LINE_AA)

    #==============================================

    # 画像の表示
    cv2.imshow("Image", frame)
 
  # `s`キーを押すと画像を保存する
    if cv2.waitKey(1) == ord('s'):
        #結果の書き込み
        cv2.imwrite('image1.jpg',frame)   
  
    # `q`キーを押すとループを終了する
    if cv2.waitKey(1) == ord('q'):
        break

# カメラを閉じる
cap.release()

# すべてのウィンドウを閉じる
cv2.destroyAllWindows()