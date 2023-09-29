def dot_product(A, B):
    # Check if the number of columns in A is equal to the number of rows in B
    if len(A[0]) != len(B):
        raise ValueError("The number of columns in the first matrix must be equal to the number of rows in the second matrix.")
    
    # Preallocate a list of zeros
    C = [[0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matrix_addition(A, B):
    # Check if the dimensions of A and B are the same
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("The dimensions of the two matrices must be the same.")
    
    # Use list comprehension for more efficient addition
    C = [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    return C

def scalar_multiplication(s, A):
    # Check if the scalar is a number
    if not isinstance(s, (int, float)):
        raise ValueError("The scalar must be a number.")
    
    # Use list comprehension for more efficient multiplication
    B = [[s * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    return B

def transpose(A):
    # Check if A is a matrix
    if not all(isinstance(row, list) for row in A):
        raise ValueError("The input must be a matrix.")
    
    # Use list comprehension for more efficient transposition
    B = [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
    return B

