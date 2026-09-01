import time
from pathlib import Path
import cv2 # need to import extra module "pip install opencv-python"
import pypuclib
from pypuclib import CameraFactory, Camera, XferData, Decoder

import static_ffmpeg
import ffmpeg

static_ffmpeg.add_paths()

BASE_DIR = Path(__file__).resolve().parent

# define max decode thread
PUC_MAX_DECODE_THREAD_COUNT = 32

print(pypuclib.__doc__)

# To connect the camera first detected
cam = CameraFactory().create()
cam.setFramerateShutter(320, 320)

# To decode image, get decoder obj from camera
decoder = cam.decoder()

# setup save video file
path = BASE_DIR / "create_movie.mp4"
vcodec = "h264"
width = 1246
height = 1008
MAX_SAVE_FRAME_COUNT = 10000
fps = 60

ffmpeg_process = None

# global variable
is_recording = False
stop_requested = False
g_count = 0         # counter of save frames
g_oldSeqNo = 0 
g_currentSeqNo = 0
start_time = 0.0

# Explanation
print("press Esc to quit this application ")
print("press 's' to start/stop saving a mp4 file")

# at first, set multi decode thread
while(1):
    threadnum = input("input decode thread num >> ")
    if threadnum.isdecimal() & (0 < int(threadnum) <= PUC_MAX_DECODE_THREAD_COUNT):
        decoder.setNumDecodeThread(int(threadnum))
        break


# callback function in transfer
def callback(data):
    global g_count
    global is_recording
    global stop_requested
    global g_oldSeqNo
    global g_currentSeqNo
    global ffmpeg_process
    global start_time
    global vcodec

    if ffmpeg_process is not None:
        src = decoder.decode(data, 0, 0, width, height)
        g_currentSeqNo = data.sequenceNo()

        if g_currentSeqNo != g_oldSeqNo:
            ffmpeg_process.stdin.write(src.tobytes())
            g_count += 1
            g_oldSeqNo = g_currentSeqNo

        if g_count >= MAX_SAVE_FRAME_COUNT or stop_requested:
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            ffmpeg_process = None
            elapsed_time = time.time() - start_time
            print(f"Encode Profile (" + str(vcodec) + "): " + f"Encode time: {elapsed_time:.2f} sec /" + f" Encode Frames: {g_count} frames /" + f" Encode AVG FPS: {g_count/elapsed_time:.2f} fps")
            g_count = 0
            is_recording = False
            stop_requested = False

# begin transfer
cam.beginXfer(callback)

while True:
    # Grab the single image data
    xferData = cam.grab()

    # Decode the data can be used as image
    img = decoder.decode(xferData)

    # get sequence number of transfer data
    seq = "sequenceNo = " + str(xferData.sequenceNo())
    cv2.putText(img, seq, (0, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2, cv2.LINE_AA)

    if is_recording:
        rec_str = f"REC: {g_count}/{MAX_SAVE_FRAME_COUNT}"
        cv2.putText(img, rec_str, (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("INFINICAM", img)

    # get key what user input
    key = cv2.waitKey(1)

    if key & 0xFF == 27: # press Esc to end application 
        break

    elif key & 0xFF == ord('s'): # press 's' to save or stop
        if not is_recording and ffmpeg_process is None:
            pix_fmt = 'gray' if len(img.shape) == 2 else 'bgr24'

            start_time = time.time()
            stop_requested = False
            ffmpeg_process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt=pix_fmt, s=f'{width}x{height}', r=fps)
                .output(str(path), vcodec=vcodec, b='8000k')
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
            is_recording = True
        elif is_recording:
            stop_requested = True


# end transfer     
if ffmpeg_process is not None:
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    elapsed_time = time.time() - start_time
    print(f"Encode Profile (" + str(vcodec) + "): " + f"Encode time: {elapsed_time:.2f} sec /" + f" Encode Frames: {g_count} frames /" + f" Encode AVG FPS: {g_count/elapsed_time:.2f} fps")

cam.endXfer()

print("end")

# Close live image window
cv2.destroyAllWindows()