#include <iostream>

// MAX THREADS PER BLOCK 64
const int maxThreadsPerBlock = 1024;
FILE *file;

#define TILE_WIDTH 32

float *readMatrixFromFile(FILE *file, int rows, int cols)
{
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

__global__ void basic_mat_mul(float *a, float *b, float *ab, int width)
{
    // calculate the row & col index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0;
    // dot product between row of a and col of b
    for (int k = 0; k < width; ++k)
        result += a[row * width + k] * b[k * width + col];
    ab[row * width + col] = result;
}

__global__ void fast_mat_mul(float *a, float *b, float *ab, int width)
{
    // shorthand
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    // allocate tiles in __shared__ memory
    __shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_b[TILE_WIDTH][TILE_WIDTH];
    // calculate the row & col index
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    float result = 0;
    // loop over the tiles of the input in phases
    for (int p = 0; p < width / TILE_WIDTH; ++p)
    {
        // collaboratively load tiles into __shared__
        s_a[ty][tx] = a[row * width + (p * TILE_WIDTH + tx)];
        s_b[ty][tx] = b[(p * TILE_WIDTH + ty) * width + col];
        __syncthreads();
        // dot product between row of s_a and col of s_b
        for (int k = 0; k < TILE_WIDTH; ++k)
            result += s_a[ty][k] * s_b[k][tx];
        __syncthreads();
    }
    ab[row * width + col] = result;
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
    basic_mat_mul<<<num_blocks, block_size>>>(d_a, d_b, d_c, MATRIX_ROW);
    fast_mat_mul<<<num_blocks, block_size>>>(d_a, d_b, d_c, MATRIX_ROW);

    // Copy data back from device to host
    cudaMemcpy(h_c, d_c, MATRIX_BYTES, cudaMemcpyDeviceToHost);

    // printMatrix(h_a, MATRIX_ROW, MATRIX_COL);
    // printMatrix(h_b, MATRIX_ROW, MATRIX_COL);

    // printMatrix(h_c, MATRIX_ROW, MATRIX_COL);

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
