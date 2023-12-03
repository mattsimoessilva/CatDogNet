def dot_product(A, B):
    """Calculate the dot product of two matrices."""
    if len(A[0]) != len(B):
        raise ValueError("The number of columns in the first matrix must be equal to the number of rows in the second matrix.")
    
    C = [[0] * len(B[0]) for _ in range(len(A))]
    for i, row_A in enumerate(A):
        for j in range(len(B[0])):
            for k, element_B in enumerate(B):
                C[i][j] += row_A[k] * element_B[j]
    return C

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
