import cv2 
import numpy as np
from pathlib import Path

# 変数、パラメータの初期設定
BASE_DIR = Path(__file__).resolve().parent

clahe = cv2.createCLAHE(clipLimit = 4.0, tileGridSize = (8, 8))

feature_params = dict(maxCorners = 2,
                      qualityLevel = 0.01,
                      minDistance = 10,
                      blockSize = 7)

subpix_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001)

lk_params = dict(winSize = (21, 21),
                 maxLevel = 3,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001))

interval = 0
total_angle = 0.0
prev_angle = None

# 画像の前処理
def process_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    return gray

# マウス左クリックによる特徴点の追加
def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

img = cv2.imread(BASE_DIR / "infinicam_coin_toss_meetingroom_10yen_1000fps" / "800.bmp")
gray_i = process_img(img)

# 特徴点選択ウィンドウの設定
window_name = 'Select Points - Click to add, \'s\':start, \'r\':reset, \'esc\':exit'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, select_points)

selecting_points = True
points = []
p0 = None

while selecting_points:
    img_show = img.copy()
    for p in points:
        cv2.circle(img_show, p, 5, (0, 0, 255), -1)

    cv2.imshow(window_name, img_show)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: # escで終了
        cv2.destroyAllWindows()
        exit()
    elif key == ord('r'): # rでリセット
        points = []
    elif key == ord('s'): # sで開始
        if len(points) == 2: # 2点選択してあると、それらを追跡
            p0 = np.array(points, dtype = np.float32).reshape(-1, 1, 2)
            selecting_points = False
        else: # それ以外の場合は範囲選択に移る
            roi = cv2.selectROI('Select Target Area', img, showCrosshair = True, fromCenter = False)
            x, y, w, h = map(int, roi)
            mask_roi = np.zeros_like(gray_i)
            mask_roi[y : y + h, x : x + w] = 255
            p0 = cv2.goodFeaturesToTrack(gray_i, mask = mask_roi, **feature_params)
            selecting_points = False

if p0 is not None:
    p0 = cv2.cornerSubPix(gray_i, p0, (17, 17), (-1, -1), subpix_criteria)

mask = np.zeros_like(img)

number_p = len(p0)
color = np.random.randint(0, 255, (number_p, 3))

# 特徴点の追跡処理    
for i in range(800, 1300):
    number = str(i) + '.bmp' 
    next_number = str(i + 1) + '.bmp' 

    img = cv2.imread(BASE_DIR / "infinicam_coin_toss_meetingroom_10yen_1000fps" / number)
    next_img = cv2.imread(BASE_DIR / "infinicam_coin_toss_meetingroom_10yen_1000fps" / next_number)

    gray_i = process_img(img)
    gray_ni = process_img(next_img)

    p1, status_f, err = cv2.calcOpticalFlowPyrLK(gray_i, gray_ni, p0, None, **lk_params)

    if p1 is not None:
        p0_b, status_b, err = cv2.calcOpticalFlowPyrLK(gray_ni, gray_i, p1, None, **lk_params)
        fb_diff = p0 - p0_b
        p1_opt = p1 + 0.5 * fb_diff

        status = (status_f.ravel() == 1) & (status_b.ravel() == 1) 
        good_new = p1_opt[status == 1]
        good_old = p0[status == 1]
    else:
        good_new = np.array([])

    if len(good_new) == 2:
        good_new = cv2.cornerSubPix(gray_ni, good_new.reshape(-1, 1, 2), (17, 17), (-1, -1), subpix_criteria).reshape(-1, 2)

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

                
    for j, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = map(int, new.ravel())
        c, d = map(int, old.ravel())

        img = cv2.circle(img, (a, b), 5, color[j].tolist(), -1)

    img = cv2.add(img, mask)    

    cv2.imshow('test', img)

    key = cv2.waitKey(interval)
    if key == 27: 
        break
    elif key == ord("s"):
        interval = 20
    elif key == ord("r"):
        interval = 0

    gray_i = gray_ni.copy()
    p0 = good_new.reshape(-1, 1, 2)

interval = 0
key = cv2.waitKey(interval)
if key == 27: 
    cv2.destroyAllWindows()