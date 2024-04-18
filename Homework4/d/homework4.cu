#include <iostream>
#include <stdio.h>

const int THREADS_PER_BLOCK = 1024;

#define MATRIX_SIZE 1024

int *readMatrixFromFile(FILE *file, int rows, int cols)
{
    int *matrix = (int *)malloc(rows * cols * sizeof(int));
    if (!matrix)
    {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (fscanf(file, "%d", &matrix[i * cols + j]) != 1)
            {
                fclose(file);
                free(matrix);
                fprintf(stderr, "Error reading matrix data\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    return matrix;
}

__global__ void findMaxNaive(const int *array, int *max)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localMax = array[tid];
    atomicMax(max, localMax);
}

__global__ void findMaxBetter(const int *array, int *max)
{
    __shared__ int localMax[THREADS_PER_BLOCK]; // Shared memory for local maxima
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    localMax[tid] = array[i];
    __syncthreads();

    // Parallel reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (localMax[tid] < localMax[tid + s])
            {
                localMax[tid] = localMax[tid + s];
            }
        }
        __syncthreads();
    }

    // First thread writes the block's maximum to global memory
    if (tid == 0)
    {
        atomicMax(max, localMax[0]);
    }
}

int main(int argc, char **argv)
{
    const char *filename = "max.txt";
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    int *h_input_matrix = readMatrixFromFile(file, MATRIX_SIZE, MATRIX_SIZE);
    fclose(file);

    int MATRIX_BYTES = MATRIX_SIZE * MATRIX_SIZE * sizeof(int);

    int *d_image_matrix;
    cudaMalloc(&d_image_matrix, MATRIX_BYTES);
    cudaMemcpy(d_image_matrix, h_input_matrix, MATRIX_BYTES, cudaMemcpyHostToDevice);

    int h_max;
    int *d_max;
    cudaMalloc(&d_max, sizeof(int));
    cudaMemcpy(d_max, &h_max, sizeof(int), cudaMemcpyHostToDevice);

    // Naive solution
    // findMaxNaive<<<(MATRIX_SIZE * MATRIX_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_image_matrix, d_max);

    cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Naive solution: Maximum value across all threads = %d\n", h_max);

    // Better solution with intermediate max values
    findMaxBetter<<<(MATRIX_SIZE * MATRIX_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_image_matrix, d_max);

    cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Better solution: Maximum value across all threads = %d\n", h_max);

    // std::cout << "Result vector:\n";
    // for (int i = 0; i < NUM_COLORS; i++)
    // {
    //     std::cout << i << ": " << h_color_counts[i] << " \n";
    // }
    std::cout << std::endl;

    cudaFree(d_image_matrix);
    cudaFree(d_max);

    free(h_input_matrix);

    return 0;
}
