import bottle
import cv2
import pypuclib
import threading
import time

print(pypuclib.__doc__)

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


frame_lock = threading.Lock()
latest_jpeg = None

shutdown_event = threading.Event()


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


def get_mjpeg_stream():
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


@bottle.route('/')
def index():
    return '''
    <html>
        <head>
            <title>Camera</title>
        </head>
        <body>
            <img id="video" width="800">
            <script>
                const video = document.getElementById('video');
                window.addEventListener('load', () => {
                    video.src = '/video_stream';
                });
                window.addEventListener('pagehide', () => {
                    video.src = '';
                });
            </script>
        </body>
    </html>
    '''


@bottle.route('/video_stream')
def video_stream():
    bottle.response.content_type = 'multipart/x-mixed-replace; boundary=frame'
    bottle.response.set_header('Cache-Control', 'no-cache, private, must-revalidate')
    bottle.response.set_header('Pragma', 'no-cache')
    bottle.response.set_header('Expires', '0')
    return get_mjpeg_stream()


if __name__ == '__main__':
    cam.beginXfer(xfer_callback)

    try:
        bottle.run(host='0.0.0.0', port=8080, server='paste')
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_event.set()
        cam.endXfer()
        if GPUStatus:
            decoder.teardownGPUDecode()
