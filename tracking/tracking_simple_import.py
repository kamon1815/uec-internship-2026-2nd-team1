import cv2 
import numpy as np
import os
import sys
import ffmpeg
import static_ffmpeg

static_ffmpeg.add_paths()

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from coin_recognition.video import Video
from coin_recognition import coin_recognition

# ディレクトリの設定
video = Video("faster_capture\\output\\1.npy")
OUTPUT_FILE = './tracking/output/tracking.mp4'

process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='gray', s=f'{video.width}x{video.height}', framerate=100)
        .output(OUTPUT_FILE, vcodec='h264_qsv')
        .overwrite_output()
        .run_async(pipe_stdin=True)
)

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

# フレームの前処理
def process_img(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(img)
    return gray

# マーカーの色（赤と緑）
color = np.array([[0, 0, 255], [0, 255, 0]])

# コマ送りのスピード（小さいほうが速い）
INTERVAL = 1

# 範囲指定の処理
bboxes = coin_recognition.get_bboxes(video)

# 最初と最後のフレーム番号
start = min(bboxes.keys()) + 10
end = max(bboxes.keys())

# 最初のフレームの設定
img_start = video.get_frame(start)
gray_i_start = process_img(img_start)

_, _, (x, y, w, h) = bboxes[start]
mask_roi = np.zeros_like(gray_i_start)
mask_roi[y : y + h, x : x + w] = 255

while True:
    # 回転数計測のための変数
    total_angle = 0.0
    prev_angle = None
    diff_angle = 0.0

    # 動画を停止状態から始める
    interval = 1

    # for i in range(start - 100, start):
    #     img = video.get_frame(i)
    #     cv2.imshow('test - \'s\':start, \'r\':stop, \'esc\':exit', img) 
    #     process.stdin.write(img.tobytes())

    #     # コマ送りの操作 
    #     key = cv2.waitKey(interval)
    #     if key == 27 or i == end: # esc:終了
    #         break
    #     elif key == ord("s"): # s:再生 
    #         interval = INTERVAL
    #     elif key == ord("r"): # r:一時停止
    #         interval = 0

    # 最初の特徴点の設定
    p0 = cv2.goodFeaturesToTrack(gray_i_start, mask = mask_roi, **feature_params)
    p0 = cv2.cornerSubPix(gray_i_start, p0, (17, 17), (-1, -1), subpix_criteria)

    # 特徴点追跡の処理    
    for i in range(start, end):
        img = video.get_frame(i)
        next_img = video.get_frame(i + 1)

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

            #cv2.putText(img, f"total_angle: {total_angle:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA) 
            cv2.putText(img, f"rotations: {abs(total_angle) / 360:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA) 
            #cv2.putText(img, f"rotations/s: {abs(diff_angle) * 1000 / 360:.2f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA) 
            
        # マーカーの表示
        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = map(int, new.ravel())
            c, d = map(int, old.ravel())

            img = cv2.circle(img, (a, b), 5, color[j].tolist(), -1)

        # フレームの表示
        cv2.imshow('test - \'s\':start, \'r\':stop, \'esc\':exit', img)
        process.stdin.write(img.tobytes())

        # コマ送りの操作 
        key = cv2.waitKey(interval)
        if key == 27 or i == end: # esc:終了
            break
        elif key == ord("s"): # s:再生 
            interval = INTERVAL
        elif key == ord("r"): # r:一時停止
            interval = 0

        # 次のフレームと特徴点の設定
        gray_i = gray_ni.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # 全ウィンドウの操作
    interval = 0
    key = cv2.waitKey(interval)
    if key == 27: # esc:終了
        break
    elif key == ord("s"): # s:再生 
        interval = INTERVAL

cv2.destroyAllWindows()
