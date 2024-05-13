#include <iostream>

// MAX THREADS PER BLOCK 64
const int maxThreadsPerBlock = 1024;
FILE *file;

#define TILE_WIDTH 32

float *readMatrixFromFile(FILE *file, int rows, int cols)
{
    float *matrix = (float *)malloc(rows * cols * sizeof(float));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
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

__global__ void basic_transpose(float *input, float *output, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < rows && y < cols)
    {
        output[y * rows + x] = input[x * cols + y];
    }
}

__global__ void fast_transpose(float *input, float *output, int rows, int cols)
{
    // __shared__ float tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH + 1]; // +1 to avoid bank conflicts

    int x_in = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y_in = blockIdx.y * TILE_WIDTH + threadIdx.y;

    int x_out = blockIdx.y * TILE_WIDTH + threadIdx.x; // Transpose block indices
    int y_out = blockIdx.x * TILE_WIDTH + threadIdx.y;

    for (int i = 0; i < TILE_WIDTH; i += blockDim.y)
    {
        if (x_in < rows && y_in + i < cols)
        {
            tile[threadIdx.y + i][threadIdx.x] = input[(y_in + i) * rows + x_in];
        }
    }

    __syncthreads();

    for (int i = 0; i < TILE_WIDTH; i += blockDim.x)
    {
        if (x_out < cols && y_out + i < rows)
        {
            output[(y_out + i) * cols + x_out] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
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

    dim3 block_size(TILE_WIDTH, TILE_WIDTH);
    dim3 num_blocks(MATRIX_ROW / block_size.x, MATRIX_ROW / block_size.y);
    int MATRIX_BYTES = MATRIX_ROW * MATRIX_COL * sizeof(float);

    // Allocate memory on the host
    float *h_a = readMatrixFromFile(file, MATRIX_ROW, MATRIX_COL);
    float *h_b = (float *)malloc(MATRIX_BYTES);

    fclose(file);

    // Allocate memory on the device
    float *d_a;
    cudaMalloc(&d_a, MATRIX_BYTES);

    float *d_b;
    cudaMalloc(&d_b, MATRIX_BYTES);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, MATRIX_BYTES, cudaMemcpyHostToDevice);

    // Launch the kernel
    basic_transpose<<<num_blocks, block_size>>>(d_a, d_b, MATRIX_ROW, MATRIX_COL);
    fast_transpose<<<num_blocks, block_size>>>(d_a, d_b, MATRIX_ROW, MATRIX_COL);

    // Copy data back from device to host
    cudaMemcpy(h_b, d_b, MATRIX_BYTES, cudaMemcpyDeviceToHost);

    // printMatrix(h_a, MATRIX_ROW, MATRIX_COL);

    // printMatrix(h_b, MATRIX_ROW, MATRIX_COL);

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);

    free(h_a);
    free(h_b);

    return 0;
}
