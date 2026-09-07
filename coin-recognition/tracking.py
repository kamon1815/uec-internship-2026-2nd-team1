import cv2 
import numpy as np
from pathlib import Path

# 変数、パラメータの初期設定
BASE_DIR = Path(__file__).resolve().parent

clahe = cv2.createCLAHE(clipLimit = 5.0, tileGridSize = (8, 8))

subpix_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001)

lk_params = dict(winSize = (21, 21),
                 maxLevel = 4,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

points = []

# 画像の前処理
def process_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    #gray = cv2.GaussianBlur(gray, (3, 3), 0)
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
        if len(points) > 0:
            p0 = np.array(points, dtype = np.float32).reshape(-1, 1, 2)
            selecting_points = False

if p0 is not None:
    p0 = cv2.cornerSubPix(gray_i, p0, (5, 5), (-1, -1), subpix_criteria)

mask = np.zeros_like(img)

number_p = len(p0)
color = np.random.randint(0, 255, (number_p, 3))

# 特徴点の追跡処理    
for i in range(800, 1400):
    number = str(i) + '.bmp' 
    next_number = str(i + 1) + '.bmp' 

    img = cv2.imread(BASE_DIR / "infinicam_coin_toss_meetingroom_10yen_1000fps" / number)
    next_img = cv2.imread(BASE_DIR / "infinicam_coin_toss_meetingroom_10yen_1000fps" / next_number)

    gray_i = process_img(img)
    gray_ni = process_img(next_img)

    p1, status, err = cv2.calcOpticalFlowPyrLK(gray_i, gray_ni, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[status == 1]
        good_old = p0[status == 1]

    for j, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = int(new[0]), int(new[1])
        c, d = int(old[0]), int(old[1])

        #mask = cv2.line(mask, (a, b), (c, d), color[j].tolist(), 2)
        img = cv2.circle(img, (a, b), 5, color[j].tolist(), -1)

    img = cv2.add(img, mask)    

    cv2.imshow('test', img)

    key = cv2.waitKey(100)
    if key == 27: 
        break

    gray_i = gray_ni.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()