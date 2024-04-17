// Included this so it is easier for me to print
#include <iostream>
// Included this because vector_add only works with C
#include <stdio.h>

// MAX THREADS PER BLOCK 64
const int maxThreadsPerBlock = 64;

__global__ void vector_add(const float *a, const float *b, float *c)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // for debugging
    printf("Thread ID: %d \n", i);
    c[i] = a[i] + b[i];
}

int main(int argc, char **argv)
{
    int VECTOR_VECTOR_BYTES;
    std::cout << "Enter the size of the vectors: ";
    std::cin >> VECTOR_VECTOR_BYTES;

    int VECTOR_BYTES = VECTOR_VECTOR_BYTES * sizeof(float);

    // allocate memory on the host
    float *h_a = (float *)malloc(VECTOR_BYTES);
    float *h_b = (float *)malloc(VECTOR_BYTES);
    float *h_c = (float *)malloc(VECTOR_BYTES);

    std::cout << "Enter elements of the first vector:\n";
    for (int i = 0; i < VECTOR_VECTOR_BYTES; i++)
    {
        std::cin >> h_a[i];
    }

    std::cout << "Enter elements of the second vector:\n";
    for (int i = 0; i < VECTOR_VECTOR_BYTES; i++)
    {
        std::cin >> h_b[i];
    }

    // allocate memory on the device
    float *d_a;
    cudaMalloc(&d_a, VECTOR_BYTES);
    float *d_b;
    cudaMalloc(&d_b, VECTOR_BYTES);
    float *d_c;
    cudaMalloc(&d_c, VECTOR_BYTES);

    // copy data from host to device
    cudaMemcpy(d_a, h_a, VECTOR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, VECTOR_BYTES, cudaMemcpyHostToDevice);

    // launch the kernel
    vector_add<<<1, maxThreadsPerBlock>>>(d_a, d_b, d_c);

    // copy data back from device to host
    cudaMemcpy(h_c, d_c, VECTOR_BYTES, cudaMemcpyDeviceToHost);

    // print the result
    std::cout << "Result vector:\n";
    for (int i = 0; i < VECTOR_VECTOR_BYTES; i++)
    {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}