import functools
import numpy as np
import pypuclib

class Video:
    def __init__(self, path):
        self.file = open(path, 'rb')
        header = np.load(self.file)
        self.framerate = header['framerate'].item()
        self.width, self.height = header['resolution'].tolist()
        self.quantization = header['quantization']
        self.frame_count = header['frame_count'].item()
        self.recorded_time = header['recorded_time'].item()
        self.recorded_fps = header['recorded_fps'].item()

        self.decoder = pypuclib.Decoder(self.quantization)
        self.reso = pypuclib.Resolution(self.width, self.height)

        self.frame_start = self.file.tell()
        np.load(self.file)
        self.frame_size = self.file.tell() - self.frame_start

    @functools.lru_cache(maxsize=32)
    def get_frame(self, index):
        self.file.seek(self.frame_start + self.frame_size * index)
        return self.decoder.decode(np.load(self.file), self.reso)

    def __del__(self):
        self.get_frame.cache_clear()
        self.file.close()
