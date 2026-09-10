from pathlib import Path
import cv2
import ffmpeg
import numpy as np
import static_ffmpeg

class Output:
    def __init__(self, width, height, path):
        self.process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', framerate=10)
            .output(path, vcodec='h264_qsv')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def write_frame(self, image):
        self.process.stdin.write(image.tobytes())

    def close(self):
        self.process.stdin.close()


def collect_points(video, background):
    points = []
    for frame in range(0, video.frame_count, INTERVAL):
        _, _, stats, centers = connected_components(video.get_frame(frame), background)
        for (cx, cy), (_, _, _, _, area) in zip(centers[1:], stats[1:]):
            if 15 <= area <= 6000 and cx > video.width * 0.5:
                points.append((frame, cx, cy))
    return np.asarray(points)


def fit_trajectory(points, video, seed=2026):
    t = points[:, 0] / video.framerate
    points_x, points_y = points[:, 1], points[:, 2]
    rng = np.random.RandomState(seed)
    best_inliers = None
    for _ in range(6000):
        first = rng.choice(len(points))
        weight = np.exp(-0.5 * ((points_y - points_y[first]) / (0.15 * video.height)) ** 2)
        weight[first] = 0
        selected = np.r_[first, rng.choice(len(points), 2, replace=False, p=weight / weight.sum())]
        selected = selected[np.argsort(t[selected])]
        if np.min(np.diff(t[selected])) < 0.05:
            continue
        cx = np.polyfit(t[selected], points_x[selected], 2)
        cy = np.polyfit(t[selected], points_y[selected], 1)
        if cx[0] >= -2 * video.width:
            continue
        distance = np.hypot(np.polyval(cx, t) - points_x, np.polyval(cy, t) - points_y)
        candidates = np.flatnonzero(distance < 14)
        candidates = candidates[np.argsort(distance[candidates])]
        _, unique = np.unique(t[candidates], return_index=True)
        inliers = candidates[unique]
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers
    if best_inliers is None:
        raise RuntimeError('コインの軌道が見つかりません')
    return (
        np.polyfit(t[best_inliers], points_x[best_inliers], 2),
        np.polyfit(t[best_inliers], points_y[best_inliers], 1),
        best_inliers,
    )


def connected_components(frame, background):
    _, dark = cv2.threshold(cv2.subtract(background, frame), 6, 255, cv2.THRESH_BINARY)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return cv2.connectedComponentsWithStats(dark)


def nearest_bbox(frame, background, center, gate=14, merge_gap=6):
    _, _, stats, centers = connected_components(frame, background)
    components = []
    for (cx, cy), (x, y, w, h, area) in zip(centers[1:], stats[1:]):
        if 15 <= area <= 6000:
            components.append((int(x), int(y), int(w), int(h), int(area), float(cx), float(cy)))
    if not components:
        return None

    px, py = center
    seed = min(components, key=lambda item: np.hypot(item[5] - px, item[6] - py))
    distance = float(np.hypot(seed[5] - px, seed[6] - py))
    if distance >= gate:
        return None

    merged = [seed]
    box = list(seed[:4])
    for component in components:
        x, y, w, h = component[:4]
        bx, by, bw, bh = box
        dx = max(bx - (x+w), x - (bx+bw), 0)
        dy = max(by - (y+h), y - (by+bh), 0)
        if component != seed and np.hypot(dx, dy) <= merge_gap:
            merged.append(component)
            x1, y1 = min(bx, x), min(by, y)
            x2, y2 = max(bx+bw, x+w), max(by+bh, y+h)
            box = [x1, y1, x2-x1, y2-y1]

    area = sum(item[4] for item in merged)
    measured = tuple(sum(item[4] * item[axis] for item in merged) / area for axis in (5, 6))
    return distance, tuple(box), measured


def track_bboxes(video, background, points, inliers, max_misses=6):
    observed = points[inliers]
    observed = observed[np.argsort(observed[:, 0])]
    rows = {}

    for frame in range(int(observed[0, 0]), int(observed[-1, 0]) + 1):
        center = tuple(np.interp(frame, observed[:, 0], observed[:, axis]) for axis in (1, 2))
        found = nearest_bbox(video.get_frame(frame), background, center)
        if found is not None:
            rows[frame] = found

    if len(rows) < 2:
        return rows
    for direction in (-1, 1):
        frames = sorted(rows)
        edge = frames[:2][::-1] if direction < 0 else frames[-2:]
        history = [(frame, *rows[frame][2]) for frame in edge]
        misses = 0
        stop = -1 if direction < 0 else video.frame_count
        for frame in range(history[-1][0] + direction, stop, direction):
            f1, x1, y1 = history[-2]
            f2, x2, y2 = history[-1]
            scale = (frame - f2) / (f2 - f1)
            center = (x2 + (x2 - x1) * scale, y2 + (y2 - y1) * scale)
            found = nearest_bbox(video.get_frame(frame), background, center)
            if found is None:
                misses += 1
                if misses >= max_misses:
                    break
                continue
            rows[frame] = found
            history.append((frame, *found[2]))
            history = history[-2:]
            misses = 0
    return rows


def recognize(video, background):
    points = collect_points(video, background)
    if points.ndim != 2 or len(points) < 3:
        return None
    try:
        cx, cy, inliers = fit_trajectory(points, video)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return None
    if inliers is None or len(inliers) < 3:
        return None
    bboxes = track_bboxes(video, background, points, inliers)
    if not bboxes:
        return None
    return cx, cy, inliers, bboxes


def get_bboxes(video):
    if video.frame_count <= 0:
        return None
    indices = np.unique(np.linspace(0, video.frame_count - 1, 25, dtype=int))
    background = np.median(
        np.stack([video.get_frame(int(frame)) for frame in indices]), axis=0
    ).astype(np.uint8)
    result = recognize(video, background)
    if result is None:
        return None
    _, _, _, rows = result
    return {
        frame: (distance, center, bbox)
        for frame, (distance, bbox, center) in rows.items()
    }


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / 'faster_capture/output/coin3.npy'
OUTPUT_FILE = ROOT / 'coin_recognition/output/coin_recognition.mp4'
INTERVAL = 1


def main():
    from video import Video

    static_ffmpeg.add_paths()
    video = Video(INPUT_FILE)

    bboxes = get_bboxes(video)
    if bboxes is None:
        print('コインを検出できませんでした')
        return None
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output = Output(video.width, video.height, str(OUTPUT_FILE))
    try:
        for frame in range(0, video.frame_count, INTERVAL):
            image = cv2.cvtColor(video.get_frame(frame), cv2.COLOR_GRAY2BGR)
            if frame in bboxes:
                _, _, (x, y, w, h) = bboxes[frame]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            output.write_frame(image)
    finally:
        output.close()

if __name__ == '__main__':
    main()

