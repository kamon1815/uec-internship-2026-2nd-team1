import cv2
import ffmpeg
import json
import numpy as np
import pathlib
import static_ffmpeg

static_ffmpeg.add_paths()
BASE_DIR = pathlib.Path('./faster_capture/output/infinicam_coin_toss_meetingroom_10yen_1000fps')

with open(BASE_DIR / 'info.json') as f:
    info_data = json.load(f)
frame_count = info_data['frame_count']

frames = []
for i in range(frame_count):
    frames.append(cv2.imread(BASE_DIR / (str(i) + '.bmp')))

print(str(pathlib.Path.cwd() / 'clean_frames' / 'output' / 'mask.mp4'))

interval = 10
height, width = frames[0].shape[:2]
process = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', framerate=interval)
    .output(str(pathlib.Path.cwd() / 'clean_frames' / 'output' / 'mask.mp4'), vcodec='h264_qsv')
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

interval = 10
for i in range(interval, frame_count, interval):
    previous = frames[i - interval]
    current = frames[i]

    previous_blur = cv2.GaussianBlur(previous, (5, 5), 0)
    current_blur = cv2.GaussianBlur(current, (5, 5), 0)

    diff = cv2.absdiff(current_blur, previous_blur)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff_gray, 7, 255, cv2.THRESH_BINARY)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))

    blob = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(opening)
    for j in range(1, count):
        x, y, blob_width, blob_height, area = map(int, stats[j])
        center_x, center_y = map(float, centroids[j])
        if not 20 <= area <= 6000:
            continue
        if not 3 <= blob_width <= 140:
            continue
        if not 3 <= blob_height <= 140:
            continue
        if not (10 < center_x < width - 10 and 10 < center_y < height * 0.92):
            continue
        cv2.rectangle(blob, (x, y), (x+blob_width, y+blob_height), (0, 255, 0), 1)

    process.stdin.write(blob.tobytes())

process.stdin.close()
process.wait()
