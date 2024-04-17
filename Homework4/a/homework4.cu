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

__global__ void vector_add(const float *a, const float *b, float *c, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // for debugging
    // printf("Thread ID: %d \n", i);
    c[i] = a[i] + b[i];
}

__global__ void vector_add_shared(const float *a, const float *b, float *c, int n)
{
    __shared__ float shared_all[1024 * 2];

    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory

    shared_all[threadIdx.x] = a[idx];
    shared_all[threadIdx.x] = b[idx];

    // Synchronize threads to ensure all data is loaded
    __syncthreads();

    // Perform vector addition using shared memory
    c[idx] = shared_all[threadIdx.x] + shared_all[threadIdx.x];
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
    float *h_b = readVectorFromFile(file, VECTOR_SIZE);
    float *h_c = (float *)malloc(VECTOR_BYTES);

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
    vector_add<<<1, maxThreadsPerBlock>>>(d_a, d_b, d_c, VECTOR_SIZE);

    // copy data back from device to host
    cudaMemcpy(h_c, d_c, VECTOR_BYTES, cudaMemcpyDeviceToHost);

    // // print the result
    // std::cout << "Result vector:\n";
    // for (int i = 0; i < VECTOR_SIZE; i++)
    // {
    //     std::cout << h_c[i] << " ";
    // }
    // std::cout << std::endl;

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}