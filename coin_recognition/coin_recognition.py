import cv2
import ffmpeg
import numpy as np
import static_ffmpeg
#from video import Video

class Output:
    def __init__(self, width, height, path):
        self.process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', framerate=10)
            .output(path, vcodec='h264_qsv')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def write_frame(self, img):
        self.process.stdin.write(img.tobytes())

    def __del__(self):
        self.process.stdin.close()
        self.process.wait()


def nearest_bbox(frame, background, center):
    _, dark = cv2.threshold(cv2.subtract(background, frame), 6, 255, cv2.THRESH_BINARY)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,  np.ones((3, 3), dtype=np.uint8))
    count, _, stats, centers = cv2.connectedComponentsWithStats(dark)
    px, py = center
    candidates = []
    for i in range(1, count):
        cx, cy = centers[i]
        x, y, w, h, area = stats[i]
        if not 15 <= area <= 6000:
            continue
        distance = np.hypot(cx - px, cy - py)
        if distance < 14:
            candidates.append((distance, (cx, cy), (x, y, w, h)))
    return min(candidates, default=None, key=lambda x: x[0])
        

def track_bboxes(video, background, points, inliers, cx, cy):
    anchor = int(points[inliers[len(inliers) // 2], 0])

    found = nearest_bbox(
        video.get_frame(anchor),
        background,
        (np.polyval(cx, anchor), np.polyval(cy, anchor))
    )

    if found is None:
        return {}

    rows = {anchor: found}

    for direction in (-1, 1):
        history = [(anchor, *found[1])]
        misses = 0

        stop = -1 if direction < 0 else video.frame_count

        for i in range(anchor + direction, stop, direction):
            if len(history) == 1:
                prediction = history[-1][1:]
            else:
                _, x1, y1 = history[-2]
                _, x2, y2 = history[-1]

                prediction = (
                    x2 + (x2 - x1),
                    y2 + (y2 - y1)
                )

            found_next = nearest_bbox(
                video.get_frame(i),
                background,
                prediction
            )

            if found_next is None:
                misses += 1
                if misses >= 6:
                    break
                continue

            rows[i] = found_next
            history.append((i, *found_next[1]))
            history = history[-2:]
            misses = 0

    return rows


def get_bboxes(video):

    # メディアン背景を作成
    indices = np.linspace(0, video.frame_count-1, 25, dtype=int)
    background = np.median(np.stack([video.get_frame(i) for i in indices]), axis=0).astype(np.uint8)
    cv2.imwrite('./coin_recognition/output/background.png', background)

    # 連結成分の重心を取得
    points = []
    interval = 10
    for i in range(0, video.frame_count, interval):
        frame = video.get_frame(i)

        # メディアン背景と比べて暗い部分を抽出
        _, dark = cv2.threshold(cv2.subtract(background, frame), 6, 255, cv2.THRESH_BINARY)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,  np.ones((3, 3), dtype=np.uint8))

        count, _, stats, centers = cv2.connectedComponentsWithStats(dark)
        for j in range(1, count):
            cx, cy = centers[j]
            _, _, _, _, area = stats[j]

            # 面積がちょうどコインに近いもの、かつ
            # 画面右半分にあるもの (ノイズ原因の手は左側にあるので全部除外してしまう、頭はどうなのか？という疑問は残る、これを入れないと上手くコインが補足されなかった)
            if 15 <= area <= 6000 and cx > video.width * 0.5:
                points.append((i, cx, cy))

    points = np.array(points)

    # 複数回試行して放物線運動を推定(RANSAC)
    t        = points[:, 0]
    points_x = points[:, 1]
    points_y = points[:, 2]

    best_inliers = None
    for _ in range(6000):

        # RANSACに使う3点を選ぶ
        # コインは比較的同じy座標を通るはずなので、y座標を優先させる
        selected_first = np.random.choice(len(points))
        omega = 0.15 * video.height
        weight = np.exp(-0.5 * ((points_y - points_y[selected_first]) / omega) ** 2)  # ガウス関数による重みづけ
        weight[selected_first] = 0                                                    # selected_first を除外
        selected = np.r_[selected_first, np.random.choice(len(points), 2, replace=False, p=weight/weight.sum())]

        # 時間間隔の短いものは近似精度が悪くなるので採用しない
        selected = selected[np.argsort(t[selected])]
        if np.min(np.diff(t[selected])) < 0.05 * video.framerate:
            continue

        # 選んだ3点から放物線軌道を作成
        cx = np.polyfit(t[selected], points_x[selected], 2)
        cy = np.polyfit(t[selected], points_y[selected], 1)

        # x方向の加速度が小さいものは除外
        if cx[0] >= -2 * video.width / video.framerate**2:
            continue

        # 時間ごとの予測位置
        predicted_x = np.polyval(cx, t)
        predicted_y = np.polyval(cy, t) 

        # 誤差
        distance = np.hypot(predicted_x - points_x, predicted_y - points_y)

        # 予測軌道との差が一定値以下のものの個数を数える(但し1フレームにつきpointは最大1つまで)
        candidates = np.flatnonzero(distance < 14)
        candidates = candidates[np.argsort(distance[candidates])]
        _, unique = np.unique(t[candidates], return_index=True)
        inliers = candidates[unique]

        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers

    cx = np.polyfit(t[best_inliers], points_x[best_inliers], 2)
    cy = np.polyfit(t[best_inliers], points_y[best_inliers], 1)

    return track_bboxes(video, background, points, best_inliers, cx, cy)


INPUT_FILE  = './faster_capture/output/infinicam_coin_toss_meetingroom_10yen_1000fps.npy'
OUTPUT_FILE = './coin_recognition/output/coin_recognition.mp4'

if __name__ == '__main__':
    static_ffmpeg.add_paths()
    video = Video(INPUT_FILE)
    output = Output(video.width, video.height, OUTPUT_FILE)

    bboxes = get_bboxes(video)

    for i in range(video.frame_count):

        show = cv2.cvtColor(video.get_frame(i), cv2.COLOR_GRAY2BGR)
        cv2.putText(show, str(i), (15, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

        if i in bboxes:
            _, _, (x,y,w,h) = bboxes[i]
            cv2.rectangle(show, (x,y), (x+w,y+h), (0,255,0), 2)

        output.write_frame(show)
