import numpy as np
from .math_operations import MODULUS_VALUE, apply_modulo_to_matrix, invert_matrix_modulo
from .file_processing import read_binary_file, parse_file_header

def collect_plain_cipher_pairs(plain_path, cipher_path, block_size):
    plain_bytes = read_binary_file(plain_path)
    cipher_bytes = read_binary_file(cipher_path)
    _, _, offset = parse_file_header(cipher_bytes)
    pairs = []
    i_p, i_c = 0, offset
    while i_p < len(plain_bytes) and i_c < len(cipher_bytes):
        plain_block = plain_bytes[i_p:i_p+block_size]
        cipher_block = cipher_bytes[i_c:i_c+block_size]
        if len(plain_block) < block_size:
            plain_block += bytes([0]*(block_size-len(plain_block)))
        if len(cipher_block) < block_size:
            cipher_block += bytes([0]*(block_size-len(cipher_block)))
        P = np.frombuffer(plain_block, dtype=np.uint8).astype(np.int64)
        C = np.frombuffer(cipher_block, dtype=np.uint8).astype(np.int64)
        pairs.append((P, C))
        i_p += block_size
        i_c += block_size
    return pairs

def recover_affine_hill_key(pairs, block_size):
    if len(pairs) < block_size+1:
        raise ValueError("need at least n+1 pairs")
    P0, C0 = pairs[0]
    DP = []
    DC = []
    for i in range(1, block_size+1):
        Pi, Ci = pairs[i]
        DP.append(apply_modulo_to_matrix(Pi - P0, MODULUS_VALUE))
        DC.append(apply_modulo_to_matrix(Ci - C0, MODULUS_VALUE))
    DP = np.stack(DP, axis=1)
    DC = np.stack(DC, axis=1)
    DP_inv = invert_matrix_modulo(DP, MODULUS_VALUE)
    if DP_inv is None:
        return None
    K = apply_modulo_to_matrix(DC.dot(DP_inv), MODULUS_VALUE)
    b = apply_modulo_to_matrix(C0 - K.dot(P0), MODULUS_VALUE)
    return K, b
