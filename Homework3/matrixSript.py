import numpy as np

def generate_matrices_file(filename, size):
    # Generate two matrices
    matrix1 = np.random.randint(0, 100, size=(size, size))
    matrix2 = np.random.randint(0, 100, size=(size, size))

    # Write matrices to file
    with open(filename, 'w') as f:
        # Write matrix dimensions
        f.write(f"{size} {size}\n" + '\n')
        
        # Write first matrix
        for row in matrix1:
            f.write(' '.join(map(str, row)) + '\n')
        
        # Separate matrices by a new line
        f.write('\n')
        
        # Write second matrix
        for row in matrix2:
            f.write(' '.join(map(str, row)) + '\n')

# Define the file name and matrix size
filename = 'Homework3/matrices.txt'
matrix_size = 8192

# Generate and write matrices to file
generate_matrices_file(filename, matrix_size)

print("Matrices generated and saved to", filename)
