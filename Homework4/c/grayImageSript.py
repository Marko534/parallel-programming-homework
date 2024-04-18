import numpy as np

def generate_matrices_file(filename, size):
    # Generate two matrices
    matrix1 = np.random.randint(0, 256, size=(size, size))

    # Write matrices to file
    with open(filename, 'w') as f:
        # Write first matrix
        for row in matrix1:
            f.write(' '.join(map(str, row)) + '\n')

# Define the file name and matrix size
filename = 'image.txt'
matrix_size = 1024

# Generate and write matrices to file
generate_matrices_file(filename, matrix_size)

print("Matrices generated and saved to", filename)
