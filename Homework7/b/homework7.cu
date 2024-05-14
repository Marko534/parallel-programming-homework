#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 1024

float *readVectorFromFile(FILE *file, int size)
{
    if (!file)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    float *vector = (float *)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++)
    {

        if (fscanf(file, "%f", &vector[i]) != 1)
        {
            fclose(file);
            fprintf(stderr, "Error reading matrix data\n");
            exit(EXIT_FAILURE);
        }
    }

    return vector;
}

__global__ void contiguous_block_sum(float *input,
                                     float *results, int n)
{
    __shared__ float sdata[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // load input into __shared__ memory
    float x = 0;
    if (i < n)
    {
        x = input[i];
    }
    sdata[threadIdx.x] = x;
    __syncthreads();
    // block-wide reduction in __shared__ mem
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            // add a partial sum upstream to our own
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        __syncthreads();
    }
    // finally, thread 0 writes the result
    if (threadIdx.x == 0)
    {
        // note that the result is per-block
        // not per-thread
        results[blockIdx.x] = sdata[0];
    }
}

__global__ void interleaved_block_sum(float *input,
                                      float *results, int n)
{
    __shared__ float sdata[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // load input into __shared__ memory
    float x = 0;
    if (i < n)
    {
        x = input[i];
    }
    sdata[threadIdx.x] = x;
    __syncthreads();

    // Perform interleaved reduction
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = 2 * stride * threadIdx.x;
        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + stride];
        }
        __syncthreads();
    }

    // Write the result
    if (threadIdx.x == 0)
    {
        results[blockIdx.x] = sdata[0];
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

    float *h_input = readVectorFromFile(file, VECTOR_LENGTH);
    float *d_input, *d_output_block, *d_output_contiguous, *d_output_interleaved;
    float result_contiguous, result_interleaved;

    // Allocate memory on GPU
    cudaMalloc((void **)&d_input, VECTOR_LENGTH * sizeof(float));
    cudaMalloc((void **)&d_output_block, num_blocks * sizeof(float));
    cudaMalloc((void **)&d_output_contiguous, sizeof(float));
    cudaMalloc((void **)&d_output_interleaved, sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, VECTOR_LENGTH * sizeof(float), cudaMemcpyHostToDevice);

    // launch one kernel to compute, per-block, a partial sum
    contiguous_block_sum<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output_block, VECTOR_LENGTH);
    contiguous_block_sum<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_output_block, d_output_contiguous, VECTOR_LENGTH);

    interleaved_block_sum<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output_block, VECTOR_LENGTH);
    interleaved_block_sum<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_output_block, d_output_interleaved, VECTOR_LENGTH);

    // Copy result back to host
    cudaMemcpy(&result_contiguous, d_output_contiguous, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result_interleaved, d_output_interleaved, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Contiguous block sum: %f\n", result_contiguous);
    printf("Interleaved block sum: %f\n", result_interleaved);

    // deallocate device memory
    cudaFree(d_input);
    cudaFree(d_output_block);
    cudaFree(d_output_contiguous);
    cudaFree(d_output_interleaved);

    // deallocate host memory
    free(h_input);

    return 0;
}
