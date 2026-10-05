__all__ = ["vector_add", "vector_subtract", "vector_sum", "scalar_multiply", "vector_mean", 
           "dot", "sum_of_squares", "magnitude", "squared_distance", "distance", "distance_magnitude", 
           "shape", "get_row", "get_column", "make_matrix", "is_diagonal", "matrix_vector_multiply", ]

import math

# somando vetores
def vector_add(v, w):
    return [v_i + w_i for v_i, w_i in zip(v, w)] if len(v) == len(w) else []

# subtraindo vetores
def vector_subtract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]

# somando lista de vetores
def vector_sum(vectors):
    result = vectors[0]
    for vector in vectors[1:]:
        result = vector_add(result, vector)
    return result

# multiplica vetor por escalar
def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

# vetor medio: tira media das componentes em cada dimensao
def vector_mean(vectors):
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

# produto escalar de dois vetores
def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

# soma dos quadrados de um vetor
def sum_of_squares(v):
    return dot(v, v)

# modulo de um vetor
def magnitude(v):
    return math.sqrt(sum_of_squares(v))

# distancia entre dois vetores
def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))

def distance(v, w):
    return math.sqrt(squared_distance(v, w))

def distance_magnitude(v, w):
    return magnitude(vector_subtract(v, w))


# matrizes
def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A and isinstance(A[0], list) else 0
    return num_rows, num_cols

def get_row(A, i):
    return A[i]

def get_column(A, j):
    return [A_i[j] for A_i in A]

def make_matrix(num_rows, num_cols, entry_fn):
    """ retorna matriz num_rows x num_cols, cuja entrada (i,j)th eh entry_fh(i,j) """
    return [[entry_fn(i, j)
             for j in range(num_cols)]
             for i in range(num_rows)]

# matriz identidade
def is_diagonal(i, j):
    return 1 if i == j else 0

def matrix_vector_multiply(matrix, vector):
    """
    Multiplica uma matriz (n x k) por um vetor (k-dimensional),
    resultando em um vetor (n-dimensional).
    """
    n_rows, k_cols = shape(matrix)
    k_elements = len(vector)

    if k_cols != k_elements:
        raise ValueError("O número de colunas da matriz deve ser igual à dimensão do vetor.")

    result_vector = [0 for i in range(n_rows)] # O vetor resultante terá n dimensões

    for i in range(n_rows):
        for j in range(k_cols):
            result_vector[i] += matrix[i][j] * vector[j]

    return result_vector

