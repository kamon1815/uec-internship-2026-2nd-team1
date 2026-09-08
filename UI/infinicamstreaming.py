import bottle
import cv2
import ngrok
import pathlib
import threading
import time
import ffmpeg
import pypuclib

BASE_DIR = pathlib.Path(__file__).resolve().parent
bottle.TEMPLATE_PATH.append(BASE_DIR / 'views')

HOST = 'localhost'
PORT = 8080
PUBLIC_URL = 'cleaver-fraction-art.ngrok-free.dev'  # https://cleaver-fraction-art.ngrok-free.dev

frame_lock = threading.Lock()
latest_jpeg = None
shutdown_event = threading.Event()

def live_stream_loop():
    while not shutdown_event.is_set():
        with frame_lock:
            frame = latest_jpeg
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

is_recording = False

@bottle.post("/toggle_recording")
def toggle_recoding():
    global is_recording
    print("ボタンが押されました")
    is_recording = not is_recording
    if is_recording:
        return {
            "recording":True
        }
    else:
        return {
            "recording":False
        }

decoder = None
reso = None
GPUStatus = None

def xfer_callback(xferData):
    global latest_jpeg
    if shutdown_event.is_set():
        return

    if GPUStatus:
        array = decoder.decodeGPU(xferData, True, reso.width)
    else:
        array = decoder.decode(xferData)
    success, encoded_image = cv2.imencode('.jpg', array)
    if success:
        with frame_lock:
            latest_jpeg = encoded_image.tobytes()


def run_server():
    bottle.run(host=HOST, port=PORT, server='waitress')


def main():
    global decoder, reso, GPUStatus

    #ngrok.forward(f'{HOST}:{PORT}', authtoken_from_env=True, domain=PUBLIC_URL)
    threading.Thread(target=run_server, daemon=True).start()


    cam = pypuclib.CameraFactory().create()
    decoder = cam.decoder()
    reso = cam.resolution()
    GPUStatus = decoder.getAvailableGPUProcess()
    if GPUStatus:
        param = pypuclib.GPUSetup(reso.width, reso.height)
        decoder.setupGPUDecode(param)
        print('Decode using a GPU device')
    else:
        print('Since GPU is not available, decode using CPU')


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

        # decoder.teardownGPUDecode()
        print("終了しました")


if __name__ == '__main__':
    main()
    