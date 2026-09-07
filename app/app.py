import bottle
import cv2
import ngrok
import pathlib
import threading
import time

BASE_DIR = pathlib.Path(__file__).resolve().parent
bottle.TEMPLATE_PATH.append(BASE_DIR / 'views')

HOST = 'localhost'
PORT = 8080
PUBLIC_URL = 'cleaver-fraction-art.ngrok-free.dev'  # https://cleaver-fraction-art.ngrok-free.dev

frame_lock = threading.Lock()
frame_latest = None
shutdown_event = threading.Event()

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

def run_server():
    bottle.run(host=HOST, port=PORT, server='waitress')

def main():
    global frame_latest

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
            time.sleep(0.03)

    except KeyboardInterrupt:
        pass

    finally:
        shutdown_event.set()
    
if __name__ == '__main__':
    main()
