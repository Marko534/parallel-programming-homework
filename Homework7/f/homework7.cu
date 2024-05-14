#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 1024
#define NUM_BINS 256

// CUDA kernel for histogram calculation using local histograms and reduction
__global__ void histogram_local_reduce(int *input, int *hist, int size)
{
    __shared__ int local_hist[NUM_BINS];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    // Initialize local histogram to 0
    if (local_tid < NUM_BINS)
    {
        local_hist[local_tid] = 0;
    }
    __syncthreads();

    // Calculate local histogram
    if (tid < size)
    {
        atomicAdd(&local_hist[(int)input[tid]], 1);
    }
    __syncthreads();

    // Reduce local histograms to global histogram
    for (int i = 0; i < NUM_BINS; i += blockDim.x)
    {
        if (local_tid + i < NUM_BINS)
        {
            atomicAdd(&hist[i + local_tid], local_hist[i + local_tid]);
        }
    }
}

int *readVectorFromFile(FILE *file, int size)
{
    if (!file)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    int *vector = (int *)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++)
    {

        if (fscanf(file, "%d", &vector[i]) != 1)
        {
            fclose(file);
            fprintf(stderr, "Error reading matrix data\n");
            exit(EXIT_FAILURE);
        }
    }

    return vector;
}

// CUDA kernel for histogram calculation using atomic operations
__global__ void histogram_atomic(int *input, int *hist, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        atomicAdd(&hist[input[tid]], 1);
    }
}

int main(void)
{

    const char *filename = "vectors.txt";
    int VECTOR_LENGTH;
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    if (fscanf(file, "%d", &VECTOR_LENGTH) != 1)
    {
        fclose(file);
        fprintf(stderr, "Error reading matrix dimensions\n");
        exit(EXIT_FAILURE);
    }

    const size_t num_blocks = (VECTOR_LENGTH / BLOCK_SIZE) + ((VECTOR_LENGTH % BLOCK_SIZE) ? 1 : 0);

    // Print the number of blocks
    printf("Number of blocks: %d\n", num_blocks);
    printf("Number of bins: %d\n", NUM_BINS);

    int *h_input = readVectorFromFile(file, VECTOR_LENGTH);
    int *d_input, *d_output_atomic, *d_output_local_reduce;

    // Allocate memory on GPU
    cudaMalloc((void **)&d_input, VECTOR_LENGTH * sizeof(int));
    cudaMalloc((void **)&d_output_atomic, NUM_BINS * sizeof(int));
    cudaMalloc((void **)&d_output_local_reduce, NUM_BINS * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, VECTOR_LENGTH * sizeof(int), cudaMemcpyHostToDevice);

    // Launch atomic histogram kernel
    histogram_atomic<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output_atomic, VECTOR_LENGTH);
    histogram_local_reduce<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output_local_reduce, VECTOR_LENGTH);

    // Copy result back to host
    int result_atomic[NUM_BINS], result_local_reduce[NUM_BINS];
    cudaMemcpy(result_atomic, d_output_atomic, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_local_reduce, d_output_local_reduce, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the atomatic results
    printf("Atomic histogram result: \n");
    for (int i = 0; i < NUM_BINS; i++)
    {
        printf("Number: %d \t Count: %d\n", i + 1, result_atomic[i]);
    }
    // Print the local reduce results
    printf("Local reduce histogram result: \n");
    for (int i = 0; i < NUM_BINS; i++)
    {
        printf("Number: %d \t Count: %d\n", i + 1, result_local_reduce[i]);
    }

    // Free memory
    free(h_input);
    cudaFree(d_input);
    cudaFree(d_output_atomic);

    return 0;
}