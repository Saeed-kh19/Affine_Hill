import numpy as np

MODULUS_VALUE = 256

def extended_euclidean_algorithm(a, b):
    if b == 0:
        return a, 1, 0
    gcd_value, x1, y1 = extended_euclidean_algorithm(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return gcd_value, x, y

def modular_inverse(a, m=MODULUS_VALUE):
    gcd_value, x, _ = extended_euclidean_algorithm(a, m)
    if gcd_value != 1:
        return None
    return x % m

def apply_modulo_to_matrix(matrix, m=MODULUS_VALUE):
    return (matrix % m).astype(np.int64)

def invert_matrix_modulo(matrix, m=MODULUS_VALUE):
    matrix = apply_modulo_to_matrix(np.array(matrix, dtype=np.int64), m)
    n = matrix.shape[0]
    identity = np.eye(n, dtype=np.int64)
    augmented = np.concatenate([matrix, identity], axis=1)

    for col in range(n):
        pivot_row = None
        for r in range(col, n):
            if extended_euclidean_algorithm(int(augmented[r, col]), m)[0] == 1:
                pivot_row = r
                break
        if pivot_row is None:
            return None
        if pivot_row != col:
            augmented[[col, pivot_row]] = augmented[[pivot_row, col]]
        inv = modular_inverse(int(augmented[col, col]), m)
        augmented[col] = apply_modulo_to_matrix(augmented[col] * inv, m)
        for r in range(n):
            if r != col:
                factor = int(augmented[r, col])
                augmented[r] = apply_modulo_to_matrix(augmented[r] - factor * augmented[col], m)
    return apply_modulo_to_matrix(augmented[:, n:], m)

def multiply_matrix_with_vector(matrix, vector, m=MODULUS_VALUE):
    return apply_modulo_to_matrix(matrix.dot(vector), m)

def add_vectors_modulo(vector_a, vector_b, m=MODULUS_VALUE):
    return apply_modulo_to_matrix(vector_a + vector_b, m)
