import cv2
import multiprocessing as mp
import numpy as np
import pypuclib
import queue
import sys
import threading
import time

HEADER_DTYPE = np.dtype([
    ('framerate',     np.uint32),
    ('shutter',       np.uint32),
    ('resolution',    np.int32, (2,)),
    ('quantization',  np.uint16, (64,)),
    ('frame_count',   np.uint64),
    ('recorded_time', np.float64),
    ('recorded_fps',  np.float64),
])

KEY_ESCAPE = 27
KEY_SPACE  = 32

FPS = 1000
WIDTH = 1246
HEIGHT = 1008
PREVIEW_FPS = 30

def preview_infinicam(width, height, quantization, frame_queue, key_queue, is_record, stop_event):
    decoder = pypuclib.Decoder(quantization)

    GPUStatus = decoder.getAvailableGPUProcess()
    if GPUStatus == True:
        param = pypuclib.GPUSetup(width, height)
        decoder.setupGPUDecode(param)
        print("Decode using a GPU device")
    elif GPUStatus == False:
        print("Since GPU is not available, decode using CPU")

    is_space_key_pressed = False
    try:
        while not stop_event.is_set():
            try:
                data = frame_queue.get_nowait()
            except queue.Empty:
                data = None

            if data is not None:
                if GPUStatus:
                    frame = decoder.decodeGPU(data, True, width)
                else:
                    frame = decoder.decode(data, pypuclib.Resolution(width, height))

                frame_show = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.putText(
                    frame_show,
                    (
                        f'press Space to {"stop" if is_record.is_set() else "start"} recording\n'
                        'press Escape to quit'
                    ),
                    (15, 35),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (0, 0, 255) if is_record.is_set() else (0, 255, 0),
                )
                if is_record.is_set():
                    cv2.rectangle(frame_show, (0, 0), (width-1, height-1), (0, 0, 255), 3)

                cv2.imshow('faster_capture_compressed', frame_show)

            key = cv2.waitKey(1) & 0xff
            if key == KEY_ESCAPE:
                key_queue.put(KEY_ESCAPE)

            if key == KEY_SPACE: 
                if not is_space_key_pressed:
                    is_space_key_pressed = True
                    key_queue.put(KEY_SPACE)
            else:
                is_space_key_pressed = False

    except KeyboardInterrupt:
        pass

    finally:
        if GPUStatus:
            decoder.teardownGPUDecode()
        cv2.destroyAllWindows()


def capture_infinicam(output_file):
    print(pypuclib.__doc__)

    cam = pypuclib.CameraFactory().create()
    cam.setFramerateShutter(FPS, FPS)
    cam.setResolution(WIDTH, HEIGHT)
    decoder = cam.decoder()
    quantization = decoder.quantization()
    reso = cam.resolution()

    ctx = mp.get_context('spawn')
    frame_queue = ctx.Queue(maxsize=1)
    frame_queue.cancel_join_thread()
    key_queue = ctx.Queue()
    is_record = ctx.Event()
    preview_stop_event = ctx.Event()

    preview_process = ctx.Process(
        target=preview_infinicam,
        args=(reso.width, reso.height, quantization, frame_queue, key_queue, is_record, preview_stop_event)
    )
    preview_process.start()

    frame_count = 0
    last_sequence_no = None
    last_preview_time = -float('inf')
    record_file = None
    record_file_lock = threading.Lock()
    shutdown_event = threading.Event()
    
    def xfer_callback(xferData):
        nonlocal frame_count, last_sequence_no, last_preview_time
        if shutdown_event.is_set():
            return
        
        data = xferData.data()

        current_time = time.perf_counter()
        if current_time - last_preview_time >= 1 / PREVIEW_FPS:
            last_preview_time = current_time
            try:
                frame_queue.put_nowait(data.copy())
            except queue.Full:
                pass

        if is_record.is_set():
            sequence_no = xferData.sequenceNo()

            with record_file_lock:
                if is_record.is_set() and sequence_no != last_sequence_no:
                    np.save(record_file, data)
                    frame_count += 1
                    last_sequence_no = sequence_no

    def start_recording():
        nonlocal frame_count, last_sequence_no, time_start, record_file

        with record_file_lock:
            frame_count = 0
            last_sequence_no = None

            record_file = open(output_file, 'wb')
            np.save(
                record_file,
                np.zeros((), dtype=HEADER_DTYPE)
            )

            time_start = time.perf_counter()
            is_record.set()

    def stop_recording():
        nonlocal record_file
        is_record.clear()
        recorded_time = time.perf_counter() - time_start

        with record_file_lock:
            recorded_fps = frame_count / recorded_time

            header = np.zeros((), dtype=HEADER_DTYPE)
            header['framerate'] = FPS
            header['shutter'] = FPS
            header['resolution'] = [reso.width, reso.height]
            header['quantization'] = quantization
            header['frame_count'] = frame_count
            header['recorded_time'] = recorded_time
            header['recorded_fps'] = recorded_fps

            record_file.seek(0)
            np.save(record_file, header)
            record_file.close()
            record_file = None

        print(
            f'frame_count:   {frame_count}\n'
            f'recorded_time: {recorded_time}\n'
            f'recorded_fps:  {recorded_fps}\n'
            'successfully saved!'
        )

    cam.beginXfer(xfer_callback)

    time_start = None
    try:
        while True:
            try:
                key = key_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if key == KEY_ESCAPE:
                break

            if key == KEY_SPACE:
                if is_record.is_set():
                    stop_recording()
                else:
                    start_recording()

    finally:
        print('exiting the program...')

        preview_stop_event.set()
        preview_process.join()

        if is_record.is_set():
            stop_recording()
        shutdown_event.set()
        cam.endXfer()


def main():
    #if len(sys.argv) < 2:
    #    print(f'usage: {__file__} <output_file>')
    #    exit(1)
    
    OUTPUT_FILE =  './faster_capture/output/coin.npy'

    try:
    #    capture_infinicam(sys.argv[1])
        capture_infinicam(OUTPUT_FILE)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
