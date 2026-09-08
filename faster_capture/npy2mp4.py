import cv2
import ffmpeg
import numpy as np
import pypuclib
import static_ffmpeg
import sys

OUTPUT_FPS = 60
SLOWDOWN = 10               # 何倍スローにするか
IS_SHOW_FRAME_COUNT = True  # フレーム数を左上に表示するか

def npy2mp4(input_file, output_file):
    with open(input_file, 'rb') as f:
        header = np.load(f)
        framerate = header['framerate'].item()
        width, height = header['resolution'].tolist()
        frame_count = header['frame_count'].item()

        decoder = pypuclib.Decoder(header['quantization'])
        reso = pypuclib.Resolution(width, height)

        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24',
                   s=f'{width}x{height}', framerate=OUTPUT_FPS)
            .output(output_file, vcodec='h264_qsv')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        for i in range(frame_count):
            data = np.load(f)

            start = round(i / framerate * OUTPUT_FPS * SLOWDOWN)
            end = round((i + 1) / framerate * OUTPUT_FPS * SLOWDOWN)
            if end <= start:
                continue

            frame = cv2.cvtColor(
                decoder.decode(data, reso),
                cv2.COLOR_GRAY2BGR
            )
            if IS_SHOW_FRAME_COUNT:
                cv2.putText(
                    frame, str(i), (15, 35),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0)
                )

            for _ in range(end - start):
                process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()


def main():
    if len(sys.argv) < 3:
        print(f'usage: {__file__} <input_file> <output_file>')
        exit(1)

    static_ffmpeg.add_paths()
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    npy2mp4(input_file, output_file)

if __name__ == '__main__':
    main()
