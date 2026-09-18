"""CRC-16-style checksum -- integer/bitwise-only, no floating point at all: 'non_fp' style."""
import numpy as np


def crc_checksum(data, out):
    crc = 0xFFFF
    for i in range(data.shape[0]):
        crc ^= data[i]
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc = crc >> 1
    out[0] = crc


KERNEL = crc_checksum
ARGS = (np.random.randint(0, 256, 4096).astype(np.int64), np.zeros(1, dtype=np.int64))
