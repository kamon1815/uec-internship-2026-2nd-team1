import cv2
import ffmpeg
import functools
import numpy as np
import pypuclib
import static_ffmpeg

class Video:
    def __init__(self, path):
        self.file = open(path, 'rb')
        header = np.load(self.file)
        self.framerate = header['framerate'].item()
        self.width, self.height = header['resolution'].tolist()
        self.quantization = header['quantization']
        self.frame_count = header['frame_count'].item()
        self.recorded_time = header['recorded_time'].item()
        self.recorded_fps = header['recorded_fps'].item()

        self.decoder = pypuclib.Decoder(self.quantization)
        self.reso = pypuclib.Resolution(self.width, self.height)

        self.frame_start = self.file.tell()
        np.load(self.file)
        self.frame_size = self.file.tell() - self.frame_start

    @functools.lru_cache(maxsize=32)
    def get_frame(self, index):
        self.file.seek(self.frame_start + self.frame_size * index)
        return self.decoder.decode(np.load(self.file), self.reso)

    def __del__(self):
        self.get_frame.cache_clear()
        self.file.close()

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

INPUT_FILE  = './faster_capture/output/infinicam_coin_toss_meetingroom_10yen_1000fps.npy'
OUTPUT_FILE = './coin_recognition/output/coin_recognition.mp4'

def main():
    static_ffmpeg.add_paths()
    video = Video(INPUT_FILE)
    output = Output(video.width, video.height, OUTPUT_FILE)

    # video.frame_count = 2600

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

        count, labels, stats, centers = cv2.connectedComponentsWithStats(dark)
        for j in range(1, count):
            cx, cy = centers[j]
            _, _, _, _, area = stats[j]

            # 面積がちょうどコインに近いもの、かつ
            # 画面右半分にあるもの (ノイズ原因の手は左側にあるので全部除外してしまう、頭はどうなのか？という疑問は残る、これを入れないと上手くコインが補足されなかった)
            if 15 <= area <= 6000 and cx > video.width * 0.5:
                points.append((i, cx, cy))

    points = np.array(points)

    # 複数回試行して放物線運動を推定(RANSAC)
    t        = points[:, 0] / video.framerate
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
        if np.min(np.diff(t[selected])) < 0.05:
            continue

        # 選んだ3点から放物線軌道を作成
        cx = np.polyfit(t[selected], points_x[selected], 2)
        cy = np.polyfit(t[selected], points_y[selected], 1)

        # x方向の加速度が小さいものは除外
        if cx[0] >= -2*video.width:
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

    for i in range(0, video.frame_count, interval):
        frame = video.get_frame(i)
        frame_show = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        t = i / video.framerate
        x = np.polyval(cx, t)
        y = np.polyval(cy, t)
        cv2.drawMarker(frame_show, (round(x), round(y)), (0, 255, 0))

        output.write_frame(frame_show)



    # interval = 10
    # for i in range(interval, video.frame_count, interval):
    #     previous = video.get_frame(i - interval)
    #     current  = video.get_frame(i)

    #     previous_blur = cv2.GaussianBlur(previous, (5, 5), 0)
    #     current_blur  = cv2.GaussianBlur(current, (5, 5), 0)

    #     diff = cv2.absdiff(current_blur, previous_blur)
    #     _, mask = cv2.threshold(diff, 7, 255, cv2.THRESH_BINARY)

    #     opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))

    #     blob = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    #     count, _, stats, centroids = cv2.connectedComponentsWithStats(opening)
    #     for j in range(1, count):
    #         x, y, w, h, area = map(int, stats[j])
    #         cv2.rectangle(blob, (x, y), (x+w, y+h), (0, 255, 0), 1)


if __name__ == '__main__':
    main()
