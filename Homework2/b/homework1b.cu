#include <iostream>

// MAX THREADS PER BLOCK 64
const int maxThreadsPerBlock = 64;

__global__ void matrix_add(const float *a, const float *b, float *c)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(int argc, char **argv)
{

    // Size of the matrix
    int MATRIX_ROW, MATRIX_COL;
    std::cout << "Enter the number of rows and columns of the matrix: ";
    std::cin >> MATRIX_ROW >> MATRIX_COL;

    int MATRIX_BYTES = MATRIX_ROW * MATRIX_COL * sizeof(float);

    // Allocate memory on the host
    float *h_a = (float *)malloc(MATRIX_BYTES);
    float *h_b = (float *)malloc(MATRIX_BYTES);
    float *h_c = (float *)malloc(MATRIX_BYTES);

    // Input the elements of the first matrix
    std::cout << "\nEnter elements of the first matrix:\n";
    for (int i = 0; i < MATRIX_ROW; i++)
    {
        for (int j = 0; j < MATRIX_COL; j++)
        {
            std::cin >> h_a[i * MATRIX_COL + j];
        }
    }

    // Input the elements of the second matrix
    std::cout << "\nEnter elements of the second matrix:\n";
    for (int i = 0; i < MATRIX_ROW; i++)
    {
        for (int j = 0; j < MATRIX_COL; j++)
        {
            std::cin >> h_b[i * MATRIX_COL + j];
        }
    }

    // Allocate memory on the device
    float *d_a;
    cudaMalloc(&d_a, MATRIX_BYTES);
    float *d_b;
    cudaMalloc(&d_b, MATRIX_BYTES);
    float *d_c;
    cudaMalloc(&d_c, MATRIX_BYTES);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, MATRIX_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, MATRIX_BYTES, cudaMemcpyHostToDevice);

    // Launch the kernel
    matrix_add<<<1, maxThreadsPerBlock>>>(d_a, d_b, d_c);

    // Copy data back from device to host
    cudaMemcpy(h_c, d_c, MATRIX_BYTES, cudaMemcpyDeviceToHost);

    // Print the result matrix
    std::cout << "\nResult matrix:\n";
    for (int i = 0; i < MATRIX_ROW; i++)
    {
        for (int j = 0; j < MATRIX_COL; j++)
        {
            std::cout << h_c[i * MATRIX_COL + j] << "\t";
        }
        std::cout << std::endl;
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
