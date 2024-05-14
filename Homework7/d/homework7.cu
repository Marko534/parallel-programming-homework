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

__global__ void glmem_reduce(float *d_in, float *d_out)
{
    int tid = threadIdx.x;
    int id = tid + blockDim.x * blockIdx.x;
    // do reduction in global mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            d_in[id] += d_in[id + s];
        }
        __syncthreads();
        // only thread 0 writes result for this block to global mem
    }
    if (tid == 0)
    {
        d_out[blockIdx.x] = d_in[id];
    }
}

__global__ void shmem_reduce(float *d_in, float *d_out, int n)
{
    __shared__ float sdata[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // load input into __shared__ memory
    float x = 0;
    if (i < n)
    {
        x = d_in[i];
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
        d_out[blockIdx.x] = sdata[0];
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
    float *d_input, *d_output_block, *d_output_global, *d_output_shared;
    float result_global, result_shared;

    // Allocate memory on GPU
    cudaMalloc((void **)&d_input, VECTOR_LENGTH * sizeof(float));
    cudaMalloc((void **)&d_output_block, num_blocks * sizeof(float));
    cudaMalloc((void **)&d_output_global, sizeof(float));
    cudaMalloc((void **)&d_output_shared, sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, VECTOR_LENGTH * sizeof(float), cudaMemcpyHostToDevice);

    // launch one kernel to compute, per-block, a partial sum
    shmem_reduce<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output_block, VECTOR_LENGTH);
    shmem_reduce<<<1, num_blocks, BLOCK_SIZE * sizeof(float)>>>(d_output_block, d_output_shared, VECTOR_LENGTH);

    glmem_reduce<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output_block);
    glmem_reduce<<<1, num_blocks>>>(d_output_block, d_output_global);

    // Copy result back to host
    cudaMemcpy(&result_shared, d_output_shared, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result_global, d_output_global, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the results of the reduction
    printf("Shared memory reduction result: %f\n", result_shared);
    printf("Global memory reduction result: %f\n", result_global);

    // deallocate device memory
    cudaFree(d_input);
    cudaFree(d_output_block);
    cudaFree(d_output_global);
    cudaFree(d_output_shared);

    // deallocate host memory
    free(h_input);

    return 0;
}
