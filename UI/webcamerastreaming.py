import bottle
import cv2
import ngrok
import pathlib
import threading
import time
import ffmpeg

BASE_DIR = pathlib.Path(__file__).resolve().parent
bottle.TEMPLATE_PATH.append(BASE_DIR / 'views')

HOST = 'localhost'
PORT = 8080
PUBLIC_URL = 'cleaver-fraction-art.ngrok-free.dev'  # https://cleaver-fraction-art.ngrok-free.dev

frame_lock = threading.Lock()
frame_latest = None
shutdown_event = threading.Event()

path = BASE_DIR / "recorded_movie.mp4"
vcodec = "h264"
MAX_SAVE_FRAME_COUNT = 10000
fps = 30
ffmpeg_process = None
is_recording = False
stop_requested = False
f_count = 0
start_time = 0.0

def live_stream_loop():
    while not shutdown_event.is_set():
        with frame_lock:
            frame = frame_latest
        if frame is not None:
            header = (
                f'--frame\r\n'
                f'Content-Type: image/jpeg\r\n'
                f'Content-Length: {len(frame)}\r\n\r\n'
            ).encode()
            yield header + frame + b'\r\n'
        time.sleep(0.03)

@bottle.get('/')
def index():
    return bottle.template('index')

@bottle.route('/live_stream')
def live_stream():
    bottle.response.content_type = 'multipart/x-mixed-replace; boundary=frame'
    bottle.response.set_header('Cache-Control', 'no-cache, private, must-revalidate')
    bottle.response.set_header('Pragma', 'no-cache')
    bottle.response.set_header('Expires', '0')
    return live_stream_loop()

@bottle.post("/toggle_recoding")
def toggle_recoding():
    global is_recording
    global stop_requested
    if not is_recording:
        is_recording = True
        stop_requested = False
        return{
            "recording":True
        }
    else:
        stop_requested = True
        return{
            "recording":False
        }


def run_server():
    bottle.run(host=HOST, port=PORT, server='waitress')

def main():
    global frame_latest
    global is_recording
    global stop_requested
    global ffmpeg_process
    global f_count
    global start_time

    ngrok.forward(f'{HOST}:{PORT}', authtoken_from_env=True, domain=PUBLIC_URL)
    threading.Thread(target=run_server, daemon=True).start()

    cap = cv2.VideoCapture(0)
    try:
        while True:
            _, frame = cap.read()
            _, frame_jpg = cv2.imencode('.jpg', frame)
            frame_bin = frame_jpg.tobytes()
            with frame_lock:
                frame_latest = frame_bin
            if is_recording and ffmpeg_process is None:
                height, width = frame.shape[:2]

                start_time = time.time()
                stop_requested = False
                f_count = 0

                ffmpeg_process = (
                    ffmpeg 
                    .input(
                        "pipe:",
                        format="rawvideo",
                        pix_fmt="bgr24", 
                        s=f"{width}*{height}",
                        r=fps
                    )
                    .output(
                        str(path),
                        vcodec=vcodec,
                        pix_fmt="yuv420p",
                        r=fps
                    )
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
                )
                print("録画開始")

            if is_recording and ffmpeg_process is not None:
                ffmpeg_process.stdin.write(frame.tobytes())
                f_count += 1

            if is_recording and (
                stop_requested or 
                f_count >= MAX_SAVE_FRAME_COUNT
            ):
                ffmpeg_process.stdin.close()
                ffmpeg_process.wait()
                print("録画終了")
                print("保存先：", path)

                ffmpeg_process = None
                is_recording = False
                stop_requested = False
                f_count = 0
            time.sleep(0.03)


    except KeyboardInterrupt:
        pass

    finally:
        shutdown_event.set()
    
if __name__ == '__main__':
    main()