// Included this so it is easier for me to print
#include <iostream>
// Included this because vector_add only works with C
#include <stdio.h>
#include <cuda_runtime.h>
// #include "../Util/Util.cuh"
#include <chrono> // for std::chrono functions

const int maxThreadsPerBlock = 1024;

class Timer
{
private:
    // Type aliases to make accessing nested type easier
    using Clock = std::chrono::steady_clock;
    using Second = std::chrono::duration<double, std::ratio<1>>;
    using Microsecond = std::chrono::duration<double, std::micro>; // Change to microseconds
    using Nanosecond = std::chrono::duration<double, std::nano>;   // Change to nanoseconds

    std::chrono::time_point<Clock> m_beg{Clock::now()};

public:
    void reset()
    {
        m_beg = Clock::now();
    }

    double elapsed() const
    {
        return std::chrono::duration_cast<Microsecond>(Clock::now() - m_beg).count();
    }
};

int *readMatrixFromFile(FILE *file, int rows, int cols)
{
    int *matrix = (int *)malloc(rows * cols * sizeof(int));

    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            if (fscanf(file, "%d", &matrix[i * cols + j]) != 1)
            {
                fclose(file);
                fprintf(stderr, "Error reading matrix data\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    return matrix;
}

void printMatrix(const int *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << matrix[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}

void transpose_CPU(int *in, int *out, int size)
{
    for (int j = 0; j < size; j++)
        for (int i = 0; i < size; i++)
            out[j + i * size] = in[i + j * size];
}

__global__ void transpose_serial(int *in, int *out, int size)
{
    for (int j = 0; j < size; j++)
        for (int i = 0; i < size; i++)
            out[j + i * size] = in[i + j * size];
    // out(j,i) = in(i,j)
}

__global__ void
transpose_parallel_per_row(int in[], int out[], int size)
{
    int i = threadIdx.x;
    for (int j = 0; j < size; j++)
        out[j + i * size] = in[i + j * size];
    // out(j,i) = in(i,j)
}

__global__ void
transpose_element_parallel(int in[], int out[], int size)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    out[j + i * size] = in[i + j * size];
    // out(j,i) = in(i,j)
}

int main(int argc, char **argv)
{
    // Size of the matrix
    int MATRIX_SIZE;
    const char *filename = "matrix.txt";

    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    if (fscanf(file, "%d %d", &MATRIX_SIZE, &MATRIX_SIZE) != 2)
    {
        fclose(file);
        fprintf(stderr, "Error reading matrix dimensions\n");
        exit(EXIT_FAILURE);
    }

    int MATRIX_BYTES = MATRIX_SIZE * MATRIX_SIZE * sizeof(int);

    // allocate memory on the host
    int *in = readMatrixFromFile(file, MATRIX_SIZE, MATRIX_SIZE);
    int *out = (int *)malloc(MATRIX_BYTES);
    int *gold = (int *)malloc(MATRIX_BYTES);

    int *d_in;
    cudaMalloc((void **)&d_in, MATRIX_BYTES);
    int *d_out;
    cudaMalloc((void **)&d_out, MATRIX_BYTES);

    // copy data to device
    cudaMemcpy(d_in, in, MATRIX_BYTES, cudaMemcpyHostToDevice);

    // Testing for cpu transpose
    Timer timer;
    timer.reset();
    transpose_CPU(in, gold, MATRIX_SIZE);
    std::cout << "CPU Time: " << timer.elapsed() << std::endl;

    // Testing for serial gpu transpose
    transpose_serial<<<1, 1>>>(d_in, d_out, MATRIX_SIZE);

    // Testing for parallel gpu transpose
    dim3 blocks0(1, 1);
    dim3 threads0(32, 1);
    transpose_parallel_per_row<<<blocks0, threads0>>>(d_in, d_out, MATRIX_SIZE);

    // Testing for element parallel gpu transpose
    dim3 blocks1(1, 1);
    dim3 threads1(32, 32);
    transpose_element_parallel<<<blocks1, threads1>>>(d_in, d_out, MATRIX_SIZE);

    // copy data to host
    cudaMemcpy(out, d_out, MATRIX_BYTES, cudaMemcpyDeviceToHost);

    // printMatrix(in, MATRIX_SIZE, MATRIX_SIZE);
    // std::cout << std::endl;
    // printMatrix(out, MATRIX_SIZE, MATRIX_SIZE);

    return 0;
}