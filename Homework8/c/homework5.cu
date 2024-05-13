#include <iostream>
#include <stdio.h>

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

__global__ void dot_product_kernel(float *x, float *y, float *dot, unsigned int n)
{
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    __shared__ float cache[maxThreadsPerBlock];

    float temp = 0.0;
    while (index < n)
    {
        temp += x[index] * y[index];

        index += stride;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();

    unsigned int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(dot, cache[0]);
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
        fprintf(stderr, "Error reading matrix dimensions\n");
        exit(EXIT_FAILURE);
    }

    float *h_vec1 = readVectorFromFile(file, VECTOR_LENGTH);
    float *h_vec2 = readVectorFromFile(file, VECTOR_LENGTH);
    float *finalResult = (float *)malloc(sizeof(float));
    fclose(file);

    int VECTOR_BYTES = VECTOR_LENGTH * sizeof(int);

    float *d_vec1, *d_vec2, *d_final;
    cudaMalloc(&d_vec1, VECTOR_BYTES);
    cudaMalloc(&d_vec2, VECTOR_BYTES);
    cudaMalloc(&d_final, sizeof(float));

    cudaMemcpy(d_vec1, h_vec1, VECTOR_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, h_vec2, VECTOR_BYTES, cudaMemcpyHostToDevice);

    dot_product_kernel<<<(VECTOR_LENGTH + maxThreadsPerBlock - 1) / maxThreadsPerBlock, maxThreadsPerBlock>>>(d_vec1, d_vec2, d_final, VECTOR_LENGTH);

    cudaMemcpy(finalResult, d_final, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Dot product: " << *finalResult << std::endl;

    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_final);

    free(h_vec1);
    free(h_vec2);
    free(finalResult);

    return 0;
}
