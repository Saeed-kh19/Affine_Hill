import numpy as np
from .math_operations import MODULUS_VALUE, multiply_matrix_with_vector, add_vectors_modulo, invert_matrix_modulo
from .file_processing import read_binary_file, write_binary_file, make_file_header, parse_file_header, split_into_blocks

def encrypt_file_with_affine_hill(input_path, output_path, key_matrix, bias_vector, block_size):
    data = read_binary_file(input_path)
    out = bytearray()
    out += make_file_header(block_size, len(data))
    for plain_block in split_into_blocks(data, block_size):
        cipher_block = add_vectors_modulo(multiply_matrix_with_vector(key_matrix, plain_block, MODULUS_VALUE), bias_vector, MODULUS_VALUE)
        out += bytes(cipher_block.astype(np.uint8))
    write_binary_file(output_path, bytes(out))

def decrypt_file_with_affine_hill(input_path, output_path, key_matrix, bias_vector):
    data = read_binary_file(input_path)
    block_size, original_length, offset = parse_file_header(data)
    key_inverse = invert_matrix_modulo(key_matrix, MODULUS_VALUE)
    if key_inverse is None:
        raise ValueError("key matrix not invertible")
    out = bytearray()
    payload = data[offset:]
    i = 0
    while i < len(payload):
        chunk = payload[i:i+block_size]
        if len(chunk) < block_size:
            chunk += bytes([0]*(block_size-len(chunk)))
        cipher_block = np.frombuffer(chunk, dtype=np.uint8).astype(np.int64)
        plain_block = multiply_matrix_with_vector(key_inverse, (cipher_block - bias_vector) % MODULUS_VALUE, MODULUS_VALUE)
        out += bytes(plain_block.astype(np.uint8))
        i += block_size
    write_binary_file(output_path, out[:original_length])
