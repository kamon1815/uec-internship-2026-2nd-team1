import time
from pathlib import Path
import cv2 # need to import extra module "pip install opencv-python"
import threading
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

# To decode image, get decoder obj from camera
decoder = cam.decoder()

# setup save video file
path = BASE_DIR / "create_movie.mp4"
vcodec = "h264"
width = 1246
height = 1008
SAVE_FRAME_COUNT = 1000
fps = 60

ffmpeg_process = None

# global variable
b_show = True      # UI flag
g_count = 0         # counter of save frames
g_oldSeqNo = 0 
g_currentSeqNo = 0
start_time = 0.0

# Explanation
print("press Esc to quit this application ")
print("press 's' to save a AVI file")

# at first, set multi decode thread
while(1):
    threadnum = input("input decode thread num >> ")
    if threadnum.isdecimal() & (0 < int(threadnum) <= PUC_MAX_DECODE_THREAD_COUNT):
        decoder.setNumDecodeThread(int(threadnum))
        break


# callback function in transfer
def callback(data):
    global g_count
    global b_show
    global g_oldSeqNo
    global g_currentSeqNo
    global ffmpeg_process
    global start_time
    global vcodec

    if g_count >= SAVE_FRAME_COUNT:
        if ffmpeg_process is not None:
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            ffmpeg_process = None
            elapsed_time = time.time() - start_time
            print(f"Encode time (" + str(vcodec) + "): " + f"{elapsed_time:.2f} sec")
        g_count = 0
        b_show = True

    if ffmpeg_process is not None:
        src = decoder.decode(data, 0, 0, width, height)
        g_currentSeqNo = data.sequenceNo()

        if g_currentSeqNo != g_oldSeqNo:
            ffmpeg_process.stdin.write(src.tobytes())
            g_count += 1
            g_oldSeqNo = g_currentSeqNo

# begin transfer
cam.beginXfer(callback)

while True:

    if b_show == True:
        # Grab the single image data
        xferData = cam.grab()

        # Decode the data can be used as image
        img = decoder.decode(xferData)

        # get sequence number of transfer data
        seq = "sequenceNo = " + str(xferData.sequenceNo())
        cv2.putText(img, seq, (0, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2, cv2.LINE_AA)

        # Show the image
        cv2.imshow("INFINICAM", img)

    # get key what user input
    key = cv2.waitKey(1)

    if key & 0xFF == 27: # press Esc to end application 
        break

    elif key & 0xFF == ord('s'): # press 's' to save avi
        b_show = False
        if ffmpeg_process is None:
            pix_fmt = 'gray' if len(img.shape) == 2 else 'bgr24'

            start_time = time.time()
            ffmpeg_process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt=pix_fmt, s=f'{width}x{height}', r=fps)
                .output(str(path), vcodec=vcodec, b='8000k')
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )


# end transfer     
if ffmpeg_process is not None:
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    elapsed_time = time.time() - start_time
    print(f"Encode time (" + str(vcodec) + "): " + f"{elapsed_time:.2f} sec")

cam.endXfer()

print("end")

# Close live image window
cv2.destroyAllWindows()