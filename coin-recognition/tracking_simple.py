import cv2 
import numpy as np
from pathlib import Path

# ディレクトリの設定
BASE_DIR = Path(__file__).resolve().parent
DIR = BASE_DIR / "infinicam_coin_toss_meetingroom_10yen_1000fps"

# 各種パラメータの設定
clahe = cv2.createCLAHE(clipLimit = 4.0, tileGridSize = (8, 8))

feature_params = dict(maxCorners = 2,
                      qualityLevel = 0.01,
                      minDistance = 10,
                      blockSize = 7)

subpix_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001)

lk_params = dict(winSize = (21, 21),
                 maxLevel = 3,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001))

# 回転数計測のための変数
total_angle = 0.0
prev_angle = None

# 最初と最後のフレーム番号
start = 800
end = 1400

# マーカーの色（赤と緑）
color = np.array([[0, 0, 255], [0, 255, 0]])

# コマ送りのスピード（1000/interval）枚/秒
interval = 0 

# フレームの前処理
def process_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    return gray

# 最初のフレームの設定
img = cv2.imread(DIR / (str(start) + '.bmp'))
gray_i = process_img(img)

# 範囲指定の処理
roi = cv2.selectROI('Select Target Area', img, showCrosshair = True, fromCenter = False)
x, y, w, h = map(int, roi)
mask_roi = np.zeros_like(gray_i)
mask_roi[y : y + h, x : x + w] = 255

# 最初の特徴点の設定
p0 = cv2.goodFeaturesToTrack(gray_i, mask = mask_roi, **feature_params)
p0 = cv2.cornerSubPix(gray_i, p0, (17, 17), (-1, -1), subpix_criteria)

# 特徴点追跡の処理    
for i in range(start, end):
    img = cv2.imread(DIR / (str(i) + '.bmp'))
    next_img = cv2.imread(DIR / (str(i + 1) + '.bmp'))

    gray_i = process_img(img)
    gray_ni = process_img(next_img)

    # 次の特徴点を推定する
    p1, status_f, err = cv2.calcOpticalFlowPyrLK(gray_i, gray_ni, p0, None, **lk_params)

    if p1 is not None: 
        # 次の特徴点を用いて今の特徴点を推定する
        p0_b, status_b, err = cv2.calcOpticalFlowPyrLK(gray_ni, gray_i, p1, None, **lk_params)
        fb_diff = p0 - p0_b
        p1_opt = p1 + 0.5 * fb_diff # 補正後の次の特徴点

        status = (status_f.ravel() == 1) & (status_b.ravel() == 1) 
        good_new = p1_opt[status == 1]
        good_old = p0[status == 1]
    else:
        good_new = np.array([])

    if len(good_new) == 2:
        good_new = cv2.cornerSubPix(gray_ni, good_new.reshape(-1, 1, 2), (17, 17), (-1, -1), subpix_criteria).reshape(-1, 2)

        # 2点を結ぶ直線と水平方向の角度を求める
        pt1 = good_new[0].ravel()
        pt2 = good_new[1].ravel()

        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]

        angle = np.degrees(np.arctan2(dy, dx))

        if prev_angle is not None:
            diff_angle = angle - prev_angle

            if diff_angle > 180:
                diff_angle -= 360
            elif diff_angle < -180:
                diff_angle += 360

            total_angle += diff_angle

        prev_angle = angle

        cv2.putText(img, f"total_angle: {total_angle:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA) 
        cv2.putText(img, f"rotations: {abs(total_angle) / 360:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA) 

    # マーカーの表示
    for j, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = map(int, new.ravel())
        c, d = map(int, old.ravel())

        img = cv2.circle(img, (a, b), 5, color[j].tolist(), -1)

    # フレームの表示
    cv2.imshow('test - \'s\':start, \'r\':stop, \'esc\':exit', img)

    # コマ送りの操作 
    key = cv2.waitKey(interval)
    if key == 27: # esc:終了
        break
    elif key == ord("s"): # s:再生 
        interval = 20
    elif key == ord("r"): # r:一時停止
        interval = 0

    # 次のフレームと特徴点の設定
    gray_i = gray_ni.copy()
    p0 = good_new.reshape(-1, 1, 2)

# 全ウィンドウの操作
interval = 0
key = cv2.waitKey(interval)
if key == 27: # esc:終了
    cv2.destroyAllWindows()