#include <iostream>
#include <stdio.h>

const int maxThreadsPerBlock = 1024;

#define NUM_COLORS 256
#define IMAGE_SIZE 1024

int *readMatrixFromFile(FILE *file, int rows, int cols)
{
    int *matrix = (int *)malloc(rows * cols * sizeof(int));
    if (!matrix)
    {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (fscanf(file, "%d", &matrix[i * cols + j]) != 1)
            {
                fclose(file);
                free(matrix);
                fprintf(stderr, "Error reading matrix data\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    return matrix;
}

__global__ void count_colors(const int *image, int *color_counts)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < IMAGE_SIZE * IMAGE_SIZE)
    {
        atomicAdd(&color_counts[image[tid]], 1);
    }
}

int main(int argc, char **argv)
{
    const char *filename = "image.txt";
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    int *h_image_matrix = readMatrixFromFile(file, IMAGE_SIZE, IMAGE_SIZE);
    fclose(file);

    int MATRIX_BYTES = IMAGE_SIZE * IMAGE_SIZE * sizeof(int);
    int *h_color_counts = (int *)calloc(NUM_COLORS, sizeof(int));
    if (!h_color_counts)
    {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    int *d_image_matrix, *d_color_counts;
    cudaMalloc(&d_image_matrix, MATRIX_BYTES);
    cudaMalloc(&d_color_counts, NUM_COLORS * sizeof(int));

    cudaMemcpy(d_image_matrix, h_image_matrix, MATRIX_BYTES, cudaMemcpyHostToDevice);

    int threadsPerBlock = maxThreadsPerBlock;
    int blocksPerGrid = (IMAGE_SIZE * IMAGE_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    count_colors<<<blocksPerGrid, threadsPerBlock>>>(d_image_matrix, d_color_counts);

    cudaMemcpy(h_color_counts, d_color_counts, NUM_COLORS * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Result vector:\n";
    for (int i = 0; i < NUM_COLORS; i++)
    {
        std::cout << i << ": " << h_color_counts[i] << " \n";
    }
    std::cout << std::endl;

    cudaFree(d_color_counts);
    cudaFree(d_image_matrix);
    free(h_color_counts);
    free(h_image_matrix);

    return 0;
}
