#include <iostream>

// MAX THREADS PER BLOCK 64
const int maxThreadsPerBlock = 1024;
FILE *file;

float *readMatrixFromFile(FILE *file, int rows, int cols)
{
    if (!file)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    float *matrix = (float *)malloc(rows * cols * sizeof(float));

    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            if (fscanf(file, "%f", &matrix[i * cols + j]) != 1)
            {
                fclose(file);
                fprintf(stderr, "Error reading matrix data\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    return matrix;
}

// Print the result matrix
void printMatrix(float *matrix, int MATRIX_ROW, int MATRIX_COL)
{
    std::cout << "\nResult matrix:\n";
    for (int i = 0; i < MATRIX_ROW; i++)
    {
        for (int j = 0; j < MATRIX_COL; j++)
        {
            std::cout << matrix[i * MATRIX_COL + j] << "\t";
        }
        std::cout << std::endl;
    }
}

__global__ void matrix_add(const float *a, const float *b, float *c)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(int argc, char **argv)
{

    // Size of the matrix
    int MATRIX_ROW, MATRIX_COL;
    const char *filename = "matrices.txt";

    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    if (fscanf(file, "%d %d", &MATRIX_ROW, &MATRIX_COL) != 2)
    {
        fclose(file);
        fprintf(stderr, "Error reading matrix dimensions\n");
        exit(EXIT_FAILURE);
    }
    int MATRIX_BYTES = MATRIX_ROW * MATRIX_COL * sizeof(float);

    // Allocate memory on the host
    float *h_a = readMatrixFromFile(file, MATRIX_ROW, MATRIX_COL);
    float *h_b = readMatrixFromFile(file, MATRIX_ROW, MATRIX_COL);
    float *h_c = (float *)malloc(MATRIX_BYTES);

    fclose(file);

    // Allocate memory on the device
    float *d_a;
    cudaMalloc(&d_a, MATRIX_BYTES);
    float *d_b;
    cudaMalloc(&d_b, MATRIX_BYTES);
    float *d_c;
    cudaMalloc(&d_c, MATRIX_BYTES);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, MATRIX_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, MATRIX_BYTES, cudaMemcpyHostToDevice);

    // Launch the kernel
    matrix_add<<<8, maxThreadsPerBlock>>>(d_a, d_b, d_c);

    // Copy data back from device to host
    cudaMemcpy(h_c, d_c, MATRIX_BYTES, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
