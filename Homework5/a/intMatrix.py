import numpy as np
import sys

def generate_matrices_file(filename, size):
    # Generate two matrices with random integer numbers within the range of sys.maxsize and -sys.maxsize - 1
    matrix1 = np.random.randint(0, 100, size=(size, size), dtype=np.int64)

    # Write matrices to file
    with open(filename, 'w') as f:
        # Write first matrix
        f.write(str(size) +'\t' + str(size)+'\n')
        for row in matrix1:
            f.write(' '.join(map(str, row)) + '\n')

# Define the file name and matrix size
filename = 'matrix.txt'
matrix_size = 32

# Generate and write matrices to file
generate_matrices_file(filename, matrix_size)

print("Matrices generated and saved to", filename)
