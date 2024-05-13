import numpy as np

def generate_vectors_file(filename, size):
    # Generate two vectors of floats between 0 and 1000
    vector1 = np.random.uniform(0, 1000, size=size)
    vector2 = np.random.uniform(0, 1000, size=size)

    # Write vectors to file
    with open(filename, 'w') as f:
        # Write vector dimensions
        f.write(f"{size}\n\n")
        
        # Write first vector
        f.write(' '.join(map(str, vector1)) + '\n')
        
        # Separate vectors by a new line
        f.write('\n')
        
        # Write second vector
        f.write(' '.join(map(str, vector2)) + '\n')

# Define the file name and vector size
filename = 'vectors.txt'
vector_size = 2048

# Generate and write vectors to file
generate_vectors_file(filename, vector_size)

print("Vectors generated and saved to", filename)
