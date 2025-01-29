#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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

__global__ void inclusive_scan(float *input, float *result)
{
    extern __shared__ float sdata[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // load input into shared memory
    float sum = (i < blockDim.x * gridDim.x) ? input[i] : 0.0f;
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1)
    {
        if (threadIdx.x >= offset)
        {
            sum += sdata[threadIdx.x - offset];
        }

        __syncthreads();
        sdata[threadIdx.x] = sum;
        __syncthreads();
    }

    if (i < blockDim.x * gridDim.x)
    {
        result[i] = sdata[threadIdx.x];
    }
}

__global__ void exclusive_scan(float *input, float *result)
{
    extern __shared__ float sdata[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    float sum = (i > 0 && i < blockDim.x * gridDim.x) ? input[i - 1] : 0.0f;
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1)
    {
        float temp;
        if (threadIdx.x >= offset)
        {
            temp = sdata[threadIdx.x - offset];
        }
        __syncthreads();
        if (threadIdx.x >= offset)
        {
            sdata[threadIdx.x] += temp;
        }
        __syncthreads();
    }

    if (i < blockDim.x * gridDim.x)
    {
        result[i] = sdata[threadIdx.x];
    }
}

int main(int argc, char **argv)
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
        fprintf(stderr, "Error reading VECTOR_LENGTH\n");
        exit(EXIT_FAILURE);
    }

    printf("Vector length: %d\n", VECTOR_LENGTH);

    const size_t num_blocks = (VECTOR_LENGTH / BLOCK_SIZE) + ((VECTOR_LENGTH % BLOCK_SIZE) ? 1 : 0);

    // Print the number of blocks
    printf("Number of blocks: %zu\n", num_blocks);

    float *h_input = readVectorFromFile(file, VECTOR_LENGTH);
    float *h_output_inclusive = (float *)malloc(VECTOR_LENGTH * sizeof(float));
    float *h_output_exclusive = (float *)malloc(VECTOR_LENGTH * sizeof(float));
    float *d_input, *d_output_inclusive, *d_output_exclusive;

    cudaMalloc(&d_input, VECTOR_LENGTH * sizeof(float));
    cudaMalloc(&d_output_inclusive, VECTOR_LENGTH * sizeof(float));
    cudaMalloc(&d_output_exclusive, VECTOR_LENGTH * sizeof(float));

    cudaMemcpy(d_input, h_input, VECTOR_LENGTH * sizeof(float), cudaMemcpyHostToDevice);

    inclusive_scan<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output_inclusive);
    exclusive_scan<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output_exclusive);

    cudaMemcpy(h_output_inclusive, d_output_inclusive, VECTOR_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_exclusive, d_output_exclusive, VECTOR_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the original vector
    printf("Original vector: \n");
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        printf("%f \t %f \t %f \t\n", h_input[i], h_output_inclusive[i], h_output_exclusive[i]);
    }

    printf("\n");

    // Free the memory
    free(h_input);
    free(h_output_inclusive);
    free(h_output_exclusive);
    cudaFree(d_input);
    cudaFree(d_output_inclusive);
    cudaFree(d_output_exclusive);
    fclose(file);
    return 0;
}
