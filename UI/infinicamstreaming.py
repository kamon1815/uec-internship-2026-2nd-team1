import bottle
import cv2
import ngrok
import pathlib
import threading
import time
import ffmpeg
import pypuclib
import tempfile
import os
import sys
import shutil


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from faster_capture import npy_viewer

from faster_capture.npy_saver import NpySaver
from tracking import tracking_UImerge

BASE_DIR = pathlib.Path(__file__).resolve().parent
bottle.TEMPLATE_PATH.append(BASE_DIR / 'views')

HOST = 'localhost'
PORT = 8080
PUBLIC_URL = 'cleaver-fraction-art.ngrok-free.dev'  # https://cleaver-fraction-art.ngrok-free.dev

frame_lock = threading.Lock()
latest_frame = None
shutdown_event = threading.Event()


def live_stream_loop():
    while not shutdown_event.is_set():
        with frame_lock:
            frame = latest_frame

        if frame is not None:
            if GPUStatus:
                array = decoder.decodeGPU(frame, True, reso.width)
            else:
                array = decoder.decode(frame, reso)

            array = cv2.rotate(array, cv2.ROTATE_90_COUNTERCLOCKWISE)

            is_success, encoded_image = cv2.imencode('.jpg', array)

            if is_success:
                jpeg = encoded_image.tobytes()

                header = (
                    f'--frame\r\n'
                    f'Content-Type: image/jpeg\r\n'
                    f'Content-Length: {len(jpeg)}\r\n\r\n'
                ).encode()

                yield header + jpeg + b'\r\n'

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

recording_lock = threading.Lock()
is_recording = False
raw_video = None

@bottle.post("/toggle_recording")
def toggle_recoding():
    global raw_video, is_recording

    print("ボタンが押されました")

    with recording_lock:
        is_recording = not is_recording
        if is_recording:
            raw_video = tempfile.NamedTemporaryFile(mode='w+b', suffix='.npy', delete=False)
            npy_saver.start_record(raw_video)

            return {
                "recording": True
            }
        else:
            npy_saver.end_record(raw_video)
            raw_video.flush()
            print("before(npy):", raw_video.name)

            output_mp4 = tempfile.NamedTemporaryFile(mode='w+b', suffix='.mp4', delete=False)
            print("after(mp4): ", output_mp4.name)            

            raw_video.seek(0)
            output_mp4.close()
            is_success, speed = tracking_UImerge.tracking(raw_video, output_mp4.name)

            print(is_success)
            raw_video.close()
            
            return {
                "recording": False,
                "output_mp4_path": pathlib.Path(output_mp4.name).name,
                "speed": speed
            }


@bottle.get('/video/<video_id>')
def video(video_id):
    return bottle.static_file(
        pathlib.Path(video_id).name,
        root=tempfile.gettempdir(),
        mimetype='video/mp4'
    )



@bottle.post('/analyze')
def analyze():
    upload = bottle.request.files.get('file')
    raw_video = tempfile.NamedTemporaryFile(
        suffix='.npy',
        delete=False
    )
    upload.save(raw_video.name, overwrite=True)
    print("before(npy):", raw_video.name)

    output_mp4 = tempfile.NamedTemporaryFile(mode='w+b', suffix='.mp4', delete=False)
    print("after(mp4): ", output_mp4.name)            

    raw_video.seek(0)
    output_mp4.close()
    is_success, speed = tracking_UImerge.tracking(raw_video, output_mp4.name)

    print(is_success)
    raw_video.close()
    
    return {
        "recording": False,
        "output_mp4_path": pathlib.Path(output_mp4.name).name,
        "speed": speed
    }




decoder = None
reso = None
GPUStatus = None

xfer_callback_count = 0
UPDATE_LATEST_FRAME_FREQUENCY = 10  # live_stream_loop の配信速度が30fpsなので、それより少し高い100fpsにする

def xfer_callback(xferData):
    global latest_frame, xfer_callback_count

    if shutdown_event.is_set():
        return

    data = xferData.data()

    with recording_lock:
        if is_recording:
            npy_saver.write_frame(raw_video, data)

    if xfer_callback_count == 0:
        with frame_lock:
            latest_frame = data.copy()

    xfer_callback_count += 1
    if xfer_callback_count % UPDATE_LATEST_FRAME_FREQUENCY == 0:
        xfer_callback_count = 0


def run_server():
    bottle.run(host=HOST, port=PORT, server='waitress')



FPS = 500
WIDTH = 1246
HEIGHT = 1024

def main():
    global decoder, reso, GPUStatus, npy_saver

    #ngrok.forward(f'{HOST}:{PORT}', authtoken_from_env=True, domain=PUBLIC_URL)
    threading.Thread(target=run_server, daemon=True).start()


    cam = pypuclib.CameraFactory().create()
    cam.setFramerateShutter(FPS, FPS)
    cam.setResolution(WIDTH, HEIGHT)
    decoder = cam.decoder()
    quantization = decoder.quantization()
    reso = cam.resolution()

    GPUStatus = decoder.getAvailableGPUProcess()
    if GPUStatus:
        param = pypuclib.GPUSetup(reso.width, reso.height)
        decoder.setupGPUDecode(param)
        print('Decode using a GPU device')
    else:
        print('Since GPU is not available, decode using CPU')

    npy_saver = NpySaver(FPS, WIDTH, HEIGHT, quantization)

    try:
        cam.beginXfer(xfer_callback)
        print("INFINICAM開始")
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass

    finally:
        print("プログラムを終了します...")

        shutdown_event.set()
        cam.endXfer()

        if GPUStatus:
            decoder.teardownGPUDecode()

        if raw_video is not None:
            raw_video.close()

        # decoder.teardownGPUDecode()
        print("終了しました")


if __name__ == '__main__':
    main()
    