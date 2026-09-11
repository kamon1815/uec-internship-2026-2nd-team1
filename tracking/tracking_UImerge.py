import cv2 
import numpy as np
import os
import sys
import tempfile
import ffmpeg
import static_ffmpeg
static_ffmpeg.add_paths()

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from coin_recognition.video import Video
from coin_recognition import multi as coin_recognition


def tracking(input_f, output_path):

    # ディレクトリの設定
    video = Video(input_f)

    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='gray', s=f'{video.height}x{video.width}', framerate=10)
        .output(output_path, vcodec='h264_qsv')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    # 各種パラメータの設定
    clahe = cv2.createCLAHE(clipLimit = 4.0, tileGridSize = (4, 4))

    feature_params = dict(
        maxCorners = 2,
        qualityLevel = 0.001,
        minDistance = 20,
        blockSize = 11
    )

    subpix_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001)

    lk_params = dict(
        winSize = (41, 41),
        maxLevel = 4,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001),
        minEigThreshold = 0.000001
    )

    # フレームの前処理
    def process_img(img):
        processed = clahe.apply(img)
        return processed

    # マーカーの色（赤と緑）
    color = np.array([[0, 0, 255], [0, 255, 0]])

    bboxes = coin_recognition.get_bboxes(video)
    if bboxes is None:
        return False, None

    # 最初と最後のフレーム番号
    filtered = [x for x in bboxes.keys() if x <= min(bboxes.keys()) + 40]
    start = max(filtered)
    end = max(bboxes.keys())

    # 最初のフレームの設定
    img_start = video.get_frame(start)
    gray_i_start = process_img(img_start)

    #roi = cv2.selectROI('Select Target Area', img_start, showCrosshair = True, fromCenter = False)
    #x, y, w, h = map(int, roi)
    _, _, (x, y, w, h) = bboxes[start]
    mask_roi = np.zeros_like(gray_i_start)
    mask_roi[y : y + h, x : x + w] = 255

    # 最初の特徴点の設定
    p0 = cv2.goodFeaturesToTrack(gray_i_start, mask = mask_roi, **feature_params)
    if p0 is None:
        return False, 0
    p0 = cv2.cornerSubPix(gray_i_start, p0, (19, 19), (-1, -1), subpix_criteria)

    # 回転数計測のための変数
    total_angle = 0.0
    prev_angle = None
    diff_angle = 0.0

    #for i in range(start - 100, start):
    #    img = video.get_frame(i)
    #    process.stdin.write(img.tobytes())

    # 特徴点追跡の処理    
    for i in range(start, end):
        img = video.get_frame(i)
        next_img = video.get_frame(i + 1)

        gray_i = process_img(img)
        gray_ni = process_img(next_img)

        # 次の特徴点を推定する
        p1, status_f, err = cv2.calcOpticalFlowPyrLK(gray_i, gray_ni, p0, None, **lk_params)

        # 特徴点の追跡に失敗したら終了
        if np.any(status_f.ravel() == 0): 
            break
        else:
            # 次の特徴点を用いて今の特徴点を推定する
            p0_b, status_b, err = cv2.calcOpticalFlowPyrLK(gray_ni, gray_i, p1, None, **lk_params)
            
            # 推定ができれば、補正を行う
            if np.any(status_b.ravel() == 0):
                p1_opt = p1
                status = (status_f.ravel() == 1)
            else:
                fb_diff = p0 - p0_b
                p1_opt = p1 + 0.5 * fb_diff
                status = (status_f.ravel() == 1) & (status_b.ravel() == 1)
            
            good_new = p1_opt[status == 1]
            good_old = p0[status == 1]          

        if len(good_new) == 2:
            good_new = cv2.cornerSubPix(gray_ni, good_new.reshape(-1, 1, 2), (19, 19), (-1, -1), subpix_criteria).reshape(-1, 2)

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

                # 角度変化で異常を検知したら終了
                if diff_angle <= 0 or diff_angle >= 100: 
                    break

                total_angle += diff_angle

            prev_angle = angle

            # cv2.putText(img, f"total_angle: {total_angle:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA) 
            #cv2.putText(img, f"rotations: {abs(total_angle) / 360:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA) 
            #cv2.putText(img, f"rotations/s: {abs(diff_angle) * 1000 / 360:.2f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA) 

        # マーカーの表示
        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = map(int, new.ravel())
            img = cv2.circle(img, (a, b), 5, color[j].tolist(), -1)

        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if len(good_new) == 2:
            cv2.putText(img, f"total_angle: {total_angle:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA) 


        process.stdin.write(img.tobytes())

        # 次のフレームと特徴点の設定
        gray_i = gray_ni.copy()
        p0 = good_new.reshape(-1, 1, 2)

    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.putText(img, f"total_angle: {total_angle:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA) 
    speed = abs(total_angle) / 360 / (i - start) * 1000
    cv2.putText(img, f"rotations/s: {speed:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA) 

    process.stdin.write(img.tobytes())
    process.stdin.close()   
    process.wait()  
    print("解析完了")

    return True, float(speed)



if __name__ == '__main__':
    OUTPUT_FILE = "faster_capture/output/coin2.npy"
    tracking(OUTPUT_FILE)