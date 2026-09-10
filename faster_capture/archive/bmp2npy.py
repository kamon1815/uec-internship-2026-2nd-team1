import cv2
import json
import numpy as np
import sys
from numba import njit, prange
from pathlib import Path


HEADER_DTYPE = np.dtype([
    ('framerate',     np.uint32),
    ('shutter',       np.uint32),
    ('resolution',    np.int32, (2,)),
    ('quantization',  np.uint16, (64,)),
    ('frame_count',   np.uint64),
    ('recorded_time', np.float64),
    ('recorded_fps',  np.float64),
])

QUANTIZATION = [
     8,  5,  5,  8, 12, 20, 25, 30,
     6,  6,  7,  9, 13, 29, 30, 27,
     7,  6,  8, 12, 20, 28, 34, 28,
     7,  8, 11, 14, 25, 43, 40, 31,
     9, 11, 18, 28, 34, 54, 51, 38,
    12, 17, 27, 32, 40, 52, 56, 46,
    24, 32, 39, 43, 51, 60, 60, 50,
    36, 46, 47, 49, 56, 50, 51, 49
]

ZIGZAG = np.array([
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
], dtype=np.int32)

BITLEN = np.zeros(32769, dtype=np.uint8)
BITLEN[1:] = (
    np.floor(np.log2(np.arange(1, 32769))) + 1
).astype(np.uint8)

C = np.array([
    [
        (1 / np.sqrt(8) if u == 0 else 0.5)
        * np.cos((2 * x + 1) * u * np.pi / 16)
        for x in range(8)
    ]
    for u in range(8)
], dtype=np.float32)


@njit(parallel=True, nogil=True, cache=True)
def encode_blocks(coeff, output):
    blocks_y, blocks_x = coeff.shape[:2]

    for k in prange(blocks_y * blocks_x):
        by = k // blocks_x
        bx = k % blocks_x

        bits = 1

        for i in range(64):
            z = ZIGZAG[i]
            value = int(coeff[by, bx, z >> 3, z & 7])
            n = int(BITLEN[abs(value)])

            bits += 1 if n == 0 else 2 * n + 1

        keep_until = 63

        if bits > 128:
            for i in range(63, 0, -1):
                z = ZIGZAG[i]
                value = int(coeff[by, bx, z >> 3, z & 7])
                n = int(BITLEN[abs(value)])

                if n:
                    bits -= 2 * n
                    keep_until = i - 1

                    if bits <= 128:
                        break

        base = by * 0xA00 + 4 + bx * 16

        output[base] = 1
        pos = 1

        for i in range(64):
            if i > keep_until:
                pos += 1
                continue

            z = ZIGZAG[i]
            value = int(coeff[by, bx, z >> 3, z & 7])

            if value == 0:
                pos += 1
                continue

            n = int(BITLEN[abs(value)])

            for _ in range(n):
                output[base + (pos >> 3)] |= 1 << (pos & 7)
                pos += 1

            pos += 1

            if value > 0:
                amplitude = value
            else:
                amplitude = (1 << n) - 1 + value

            for bit in range(n - 1, -1, -1):
                if amplitude & (1 << bit):
                    output[base + (pos >> 3)] |= 1 << (pos & 7)

                pos += 1


def compress_frame(frame, inv_q):
    height, width = frame.shape
    padded_width = (width + 7) // 8 * 8

    padded = np.empty(
        (height, padded_width),
        dtype=np.float32
    )

    padded[:, :width] = frame

    if padded_width > width:
        padded[:, width:] = frame[:, -1, None]

    padded -= 128

    blocks = (
        padded
        .reshape(
            height // 8,
            8,
            padded_width // 8,
            8
        )
        .transpose(0, 2, 1, 3)
    )

    dct = C @ blocks @ C.T
    coeff = np.rint(dct * inv_q).astype(np.int16)

    output = np.zeros(
        height * 320,
        dtype=np.uint8
    )

    encode_blocks(coeff, output)

    return output


def convert(input_dir, output_file):
    input_dir = Path(input_dir)

    with open(input_dir / 'info.json') as f:
        old_info = json.load(f)

    files = sorted(
        input_dir.glob('*.bmp'),
        key=lambda p: int(p.stem)
    )

    frame_count = old_info['frame_count']
    recorded_time = old_info['time']
    recorded_fps = old_info['fps']

    if len(files) != frame_count:
        raise ValueError(
            f'frame_count={frame_count}, bmp files={len(files)}'
        )

    first = cv2.imread(
        str(files[0]),
        cv2.IMREAD_GRAYSCALE
    )

    height, width = first.shape

    if height % 8:
        raise ValueError('height must be divisible by 8')

    q = np.array(
        QUANTIZATION,
        dtype=np.float32
    ).reshape(8, 8)

    inv_q = 1.0 / q

    # 仮設定:
    # 過去のinfo.jsonには指定framerate/shutterが残っていないため、
    # recorded_fpsが500または1000の近い方だったと仮定する。
    framerate = min(
        (500, 1000),
        key=lambda x: abs(recorded_fps - x)
    )
    shutter = framerate

    print(
        f'framerate = {framerate}, shutter = {shutter} '
        '(recorded_fpsから500/1000の近い方を仮設定)'
    )

    header = np.zeros((), dtype=HEADER_DTYPE)
    header['framerate'] = framerate
    header['shutter'] = shutter
    header['resolution'] = [width, height]
    header['quantization'] = QUANTIZATION
    header['frame_count'] = frame_count
    header['recorded_time'] = recorded_time
    header['recorded_fps'] = recorded_fps

    # Numbaを先にコンパイルして、進捗表示開始後の初回だけ遅くなるのを防ぐ
    compress_frame(first, inv_q)

    with open(output_file, 'wb') as f:
        np.save(f, header)

        for i, file in enumerate(files):
            frame = cv2.imread(
                str(file),
                cv2.IMREAD_GRAYSCALE
            )

            np.save(
                f,
                compress_frame(frame, inv_q)
            )

            if (i + 1) % 100 == 0 or i + 1 == frame_count:
                print(f'{i + 1} / {frame_count}')


def main():
    if len(sys.argv) < 3:
        print(
            f'usage: {__file__} '
            '<input_dir> <output_file>'
        )
        exit(1)

    convert(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()

