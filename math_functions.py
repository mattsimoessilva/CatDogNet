def dot_product(matrix_A, matrix_B):
    """
    Computes the dot product of two matrices.
    Assumes the matrices are valid for multiplication.

    Parameters:
        matrix_A (list): First matrix.
        matrix_B (list): Second matrix.

    Returns:
        list: Resulting matrix.
    """
    num_rows_A, num_cols_A = len(matrix_A), len(matrix_A[0])
    num_rows_B, num_cols_B = len(matrix_B), len(matrix_B[0])

    # Check if the matrices are valid for multiplication
    if num_cols_A != num_rows_B:
        raise ValueError("The number of columns in the first matrix must be equal to the number of rows in the second matrix.")

    result = [[0.0] * num_cols_B for _ in range(num_rows_A)]

    for i in range(num_rows_A):
        for j in range(num_cols_B):
            for k in range(num_cols_A):
                result[i][j] += matrix_A[i][k] * matrix_B[k][j]

    return result

def matrix_addition(A, B):
    """Add two matrices element-wise."""
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("The dimensions of the two matrices must be the same.")
    
    C = [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    return C

def scalar_multiplication(s, A):
    """Multiply a matrix by a scalar."""
    if not isinstance(s, (int, float)):
        raise ValueError("The scalar must be a number.")
    
    B = [[s * element for element in row] for row in A]
    return B

def transpose(A):
    """Transpose a matrix."""
    if not all(isinstance(row, list) and len(row) == len(A[0]) for row in A):
        raise ValueError("The input must be a matrix.")
    
    B = [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
    return B
