import cv2
import json
from pathlib import Path
import pypuclib
import os
import shutil
import threading
import time


infinicam_frame_lock = threading.Lock()
infinicam_frame_latest = None
infinicam_frame_count = 0
infinicam_shutdown_event = threading.Event()

infinicam_recorded_data = []
infinicam_last_sequence_no = None
infinicam_last_preview_time = 0


def infinicam_xfer_callback(xferData, decoder, reso, GPUStatus, is_record, output_dir):
    global infinicam_frame_latest, infinicam_frame_count
    global infinicam_last_sequence_no, infinicam_last_preview_time

    if infinicam_shutdown_event.is_set():
        return

    if is_record:
        sequence_no = xferData.sequenceNo()

        if sequence_no != infinicam_last_sequence_no:
            # 圧縮された転送データだけをRAMへ保存
            infinicam_recorded_data.append(xferData.data().copy())
            infinicam_frame_count += 1
            infinicam_last_sequence_no = sequence_no

        return

    # プレビューは約30fpsに制限
    current_time = time.perf_counter()
    if current_time - infinicam_last_preview_time < 1 / 30:
        return

    infinicam_last_preview_time = current_time

    if GPUStatus:
        frame = decoder.decodeGPU(xferData, True, reso.width)
    else:
        frame = decoder.decode(xferData)

    with infinicam_frame_lock:
        infinicam_frame_latest = frame


def capture_infinicam(output_dir):
    global infinicam_frame_count
    global infinicam_recorded_data
    global infinicam_last_sequence_no
    global infinicam_last_preview_time

    print(pypuclib.__doc__)

    FPS = 1000
    WIDTH = 1246
    HEIGHT = 1008

    infinicam_frame_count = 0
    infinicam_recorded_data = []
    infinicam_last_sequence_no = None
    infinicam_last_preview_time = 0
    infinicam_shutdown_event.clear()

    cam = pypuclib.CameraFactory().create()
    cam.setFramerateShutter(FPS, FPS)
    cam.setResolution(WIDTH, HEIGHT)

    decoder = cam.decoder()
    reso = cam.resolution()

    GPUStatus = decoder.getAvailableGPUProcess()

    if GPUStatus:
        param = pypuclib.GPUSetup(reso.width, reso.height)
        decoder.setupGPUDecode(param)
        print('Decode using a GPU device')
    else:
        decoder.setNumDecodeThread(min(os.cpu_count() or 1, 32))
        print(
            'Since GPU is not available, decode using CPU '
            f'({min(os.cpu_count() or 1, 32)} threads)'
        )

    is_record = False

    cam.beginXfer(
        lambda xferData:
        infinicam_xfer_callback(
            xferData,
            decoder,
            reso,
            GPUStatus,
            is_record,
            output_dir
        )
    )

    is_s_key_pressed = False
    time_start = None
    time_stop = None

    try:
        while True:
            with infinicam_frame_lock:
                frame = infinicam_frame_latest

            if frame is not None:
                frame_show = frame.copy()

                cv2.putText(
                    frame_show,
                    f'Press S to {"stop" if is_record else "start"} recording',
                    org=(15, 35),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1,
                    color=0,
                )

                cv2.imshow('mode 1: capture infinicam', frame_show)

            if cv2.waitKey(1) == ord('s') and not is_s_key_pressed:
                is_s_key_pressed = True

                if is_record:
                    is_record = False
                    time_stop = time.perf_counter()
                    break
                else:
                    infinicam_frame_count = 0
                    infinicam_recorded_data.clear()
                    infinicam_last_sequence_no = None

                    time_start = time.perf_counter()
                    is_record = True

            else:
                is_s_key_pressed = False

    except KeyboardInterrupt:
        if is_record:
            is_record = False
            time_stop = time.perf_counter()

    finally:
        infinicam_shutdown_event.set()
        cam.endXfer()

        if time_start is not None and time_stop is not None:
            time_interval = time_stop - time_start

            info = {
                'frame_count': infinicam_frame_count,
                'time': time_interval,
                'fps': infinicam_frame_count / time_interval
            }

            with open(Path(output_dir) / 'info.json', 'w') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)

            print(
                f'Captured {infinicam_frame_count} frames '
                f'in {time_interval:.3f} s '
                f'({infinicam_frame_count / time_interval:.1f} fps)'
            )

        if infinicam_recorded_data:
            print('Decoding and saving frames...')

            for i, xferData in enumerate(infinicam_recorded_data):
                if GPUStatus:
                    frame = decoder.decodeGPU(
                        xferData,
                        True,
                        reso.width
                    )
                else:
                    frame = decoder.decode(
                        xferData,
                        reso
                    )

                cv2.imwrite(
                    Path(output_dir) / f'{i}.bmp',
                    frame
                )

                if (i + 1) % 100 == 0:
                    print(
                        f'{i + 1} / '
                        f'{len(infinicam_recorded_data)}'
                    )

        infinicam_recorded_data.clear()

        if GPUStatus:
            decoder.teardownGPUDecode()

        cv2.destroyAllWindows()


def capture_web_camera(output_dir):
    cap = cv2.VideoCapture(0)
    frame_count = 0
    is_record = False
    is_s_key_pressed = False

    try:
        while True:
            success, frame = cap.read()

            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_show = frame.copy()

                cv2.putText(
                    frame_show,
                    f'Press S to {"stop" if is_record else "start"} recording',
                    org=(15, 35),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1,
                    color=0,
                )

                cv2.imshow(
                    'mode 2: capture web camera',
                    frame_show
                )

                if is_record:
                    cv2.imwrite(
                        Path(output_dir) / f'{frame_count}.bmp',
                        frame
                    )
                    frame_count += 1

            if cv2.waitKey(1) == ord('s') and not is_s_key_pressed:
                is_s_key_pressed = True

                if is_record:
                    time_interval = time.time() - time_start

                    info = {
                        'frame_count': frame_count,
                        'time': time_interval,
                        'fps': frame_count / time_interval
                    }

                    with open(Path(output_dir) / 'info.json', 'w') as f:
                        json.dump(
                            info,
                            f,
                            ensure_ascii=False,
                            indent=2
                        )

                    break

                else:
                    is_record = True
                    time_start = time.time()

            else:
                is_s_key_pressed = False

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':

    print(
        'select mode:\r\n'
        '\t1: capture infinicam\r\n'
        '\t2: capture web camera'
    )

    try:
        while True:
            mode = input()

            if mode == '1' or mode == '2':
                print('enter the output directory:')
                output_dir = input()

                if os.path.isdir(output_dir):
                    shutil.rmtree(output_dir)

                os.makedirs(output_dir, exist_ok=True)

                if mode == '1':
                    capture_infinicam(output_dir)

                elif mode == '2':
                    capture_web_camera(output_dir)

                break

            else:
                print(
                    'This mode does not exist. '
                    'Please enter it again...'
                )

    except KeyboardInterrupt:
        pass
