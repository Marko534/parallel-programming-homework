#include <stdio.h>

#define N 16
#define THREADS_PER_BLOCK 8

__global__ void sum(int *input, int *output)
{
    __shared__ int partialSum[THREADS_PER_BLOCK];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    partialSum[threadIdx.x] = input[i] + input[i + THREADS_PER_BLOCK];
    __syncthreads();

    if (threadIdx.x < 4)
    {
        partialSum[threadIdx.x] += partialSum[threadIdx.x + 4];
        __syncthreads();
    }

    if (threadIdx.x < 2)
    {
        partialSum[threadIdx.x] += partialSum[threadIdx.x + 2];
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        *output = partialSum[0] + partialSum[1];
    }
}

int main()
{
    int *d_input, *d_output;
    int input[N];
    int result = 0;

    // Initialize input data
    for (int i = 0; i < N; ++i)
    {
        input[i] = i + 1;
    }

    // Allocate memory on GPU
    cudaMalloc((void **)&d_input, N * sizeof(int));
    cudaMalloc((void **)&d_output, sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with multiple blocks
    sum<<<1, THREADS_PER_BLOCK>>>(d_input, d_output);

    // Copy result back to host
    cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sum of 16 numbers: %d\n", result);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
