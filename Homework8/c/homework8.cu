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

__global__ void blelloch_scan(float *input, float *output, int n)
{
    extern __shared__ float temp[]; // shared memory

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory and perform reduction (up-sweep) phase
    if (idx < n)
    {
        temp[tid] = input[idx];

        for (int d = 1; d < blockDim.x; d *= 2)
        {
            int ai = tid - d;
            if (ai >= 0)
            {
                temp[tid] += temp[ai];
            }
            __syncthreads();
        }
    }

    // Clear the last element
    if (tid == 0 && idx == n - 1)
    {
        temp[n - 1] = 0;
    }

    // Perform post-reduction (down-sweep) phase
    for (int d = blockDim.x / 2; d > 0; d /= 2)
    {
        __syncthreads();
        int ai = tid - d;
        if (ai >= 0)
        {
            float t = temp[ai];
            temp[ai] = temp[tid];
            temp[tid] += t;
        }
    }
    __syncthreads();

    // Write the result to output
    if (idx < n)
    {
        output[idx] = temp[tid];
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

    blelloch_scan<<<num_blocks, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, VECTOR_LENGTH);

    cudaMemcpy(h_output, d_output, VECTOR_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Original vector and its inclusive scan result:\n");
    for (int i = 0; i < VECTOR_LENGTH; i++)
    {
        printf("%f\t%f\n", h_input[i], h_output[i]);
    }
    // printf("\n");

    // Free the memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    fclose(file);
    return 0;
}
