// Included this so it is easier for me to print
#include <iostream>
// Included this because vector_add only works with C
#include <stdio.h>
// #include "../Util/Util.cuh"

const int maxThreadsPerBlock = 1024;

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

__global__ void adjacent_difference_kernel(const float *input, float *output, int n)
{
    // Define shared memory
    __shared__ float shared_data[1024];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Load input into shared memory
    if (idx < n)
    {
        shared_data[tid] = input[idx];
    }

    __syncthreads(); // Ensure all data is loaded into shared memory

    // Calculate adjacent difference
    if (tid < blockDim.x - 1 && idx < n - 1)
    {
        output[idx] = shared_data[tid + 1] - shared_data[tid];
    }
}

int main(int argc, char **argv)
{
    // Size of the matrix
    int VECTOR_SIZE;
    const char *filename = "vectors.txt";

    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    if (fscanf(file, "%d", &VECTOR_SIZE) != 1)
    {
        fclose(file);
        fprintf(stderr, "Error reading matrix dimensions\n");
        exit(EXIT_FAILURE);
    }

    int VECTOR_BYTES = VECTOR_SIZE * sizeof(float);

    // allocate memory on the host
    float *h_a = readVectorFromFile(file, VECTOR_SIZE);
    float *h_c = (float *)malloc(VECTOR_BYTES);

    // allocate memory on the device
    float *d_a;
    cudaMalloc(&d_a, VECTOR_BYTES);
    float *d_c;
    cudaMalloc(&d_c, VECTOR_BYTES);

    // copy data from host to device
    cudaMemcpy(d_a, h_a, VECTOR_BYTES, cudaMemcpyHostToDevice);

    // launch the kernel
    adjacent_difference_kernel<<<1, maxThreadsPerBlock>>>(d_a, d_c, VECTOR_SIZE);

    // copy data back from device to host
    cudaMemcpy(h_c, d_c, VECTOR_BYTES, cudaMemcpyDeviceToHost);

    // print the result
    std::cout << "Result vector:\n";
    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // free memory
    cudaFree(d_a);
    cudaFree(d_c);

    free(h_a);
    free(h_c);

    return 0;
}