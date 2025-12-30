import struct
import numpy as np

MAGIC_TAG = b"AHIL"
VERSION_NUMBER = 1

def read_binary_file(path):
    with open(path, "rb") as f:
        return f.read()

def write_binary_file(path, data):
    with open(path, "wb") as f:
        f.write(data)

def make_file_header(block_size, original_length):
    return MAGIC_TAG + bytes([VERSION_NUMBER]) + bytes([block_size]) + struct.pack("<Q", original_length)

def parse_file_header(data):
    if data[:4] != MAGIC_TAG:
        raise ValueError("bad header")
    block_size = data[5]
    original_length = struct.unpack("<Q", data[6:14])[0]
    return block_size, original_length, 14

def split_into_blocks(buffer, block_size):
    i = 0
    while i < len(buffer):
        chunk = buffer[i:i+block_size]
        if len(chunk) < block_size:
            chunk += bytes([0]*(block_size-len(chunk)))
        yield np.frombuffer(chunk, dtype=np.uint8).astype(np.int64)
        i += block_size
