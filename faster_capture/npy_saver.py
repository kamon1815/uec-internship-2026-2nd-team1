import numpy as np
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

class NpySaver:

    def __init__(self, framerate, width, height, quantization):
        self.framerate = framerate
        self.width  = width
        self.height = height
        self.quantization = quantization

    def start_record(self, f):
        np.save(f, np.zeros((), dtype=HEADER_DTYPE))
        self.time_start = time.perf_counter()
        self.frame_count = 0

    def write_frame(self, f, frame_bin):
        np.save(f, frame_bin)
        self.frame_count += 1


    def end_record(self, f):
        recorded_time = time.perf_counter() - self.time_start
        recorded_fps = self.frame_count / recorded_time
        
        header = np.zeros((), dtype=HEADER_DTYPE)
        header['framerate'] = self.framerate
        header['shutter'] = self.framerate
        header['resolution'] = [self.width, self.height]
        header['quantization'] = self.quantization
        header['frame_count'] = self.frame_count
        header['recorded_time'] = recorded_time
        header['recorded_fps'] = recorded_fps

        f.seek(0)
        np.save(f, header)
