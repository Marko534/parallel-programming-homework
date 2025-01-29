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

__global__ void hillis_steele_scan(float *input, float *output, int n)
{
    extern __shared__ float temp[]; // shared memory

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = 1;

    if (idx < n)
    {
        temp[2 * tid] = input[2 * tid];
        temp[2 * tid + 1] = input[2 * tid + 1];
    }

    // Up-sweep (reduce) phase
    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d && idx + d < n)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (tid == 0)
        temp[n - 1] = 0;

    // Down-sweep (scan) phase
    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (tid < d && idx + d < n)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    // Write the result to output
    if (idx < n)
    {
        output[2 * tid] = temp[2 * tid];
        output[2 * tid + 1] = temp[2 * tid + 1];
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

    float *h_input = readVectorFromFile(file, VECTOR_LENGTH);
    float *h_output = (float *)malloc(VECTOR_LENGTH * sizeof(float));
    float *d_input, *d_output;

    cudaMalloc(&d_input, VECTOR_LENGTH * sizeof(float));
    cudaMalloc(&d_output, VECTOR_LENGTH * sizeof(float));

    cudaMemcpy(d_input, h_input, VECTOR_LENGTH * sizeof(float), cudaMemcpyHostToDevice);

    hillis_steele_scan<<<num_blocks, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, VECTOR_LENGTH);

    cudaMemcpy(h_output, d_output, VECTOR_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Original vector and its inclusive scan result:\n");
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        printf("%f\t%f\n", h_input[i], h_output[i]);
    }
    printf("\n");

    // Free the memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    fclose(file);
    return 0;
}
