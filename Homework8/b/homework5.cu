#include <iostream>
#include <fstream>

const int DIM = 8192 * 2; // Define the dimensions of the image

struct cuComplex
{
    float r;
    float i;
    __device__ cuComplex(float a, float b) : r(a), i(b) {}
    __device__ float magnitude2(void) { return r * r + i * i; }
    __device__ cuComplex operator*(const cuComplex &a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ cuComplex operator+(const cuComplex &a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);
    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);
    int i = 0;
    for (i = 0; i < 200; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return i; // Return the number of iterations taken to escape
    }
    return 200; // Return a constant value if no escape occurs
}

__global__ void kernel(unsigned char *ptr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < DIM && y < DIM)
    {
        int offset = x + y * DIM;
        int juliaValue = julia(x, y);

        // Map the iteration count to a color gradient
        float t = (float)juliaValue / 200; // 200 is the maximum number of iterations
        unsigned char r = (unsigned char)(9 * (1 - t) * t * t * t * 255);
        unsigned char g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
        unsigned char b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);

        // Assign the calculated color to the pixel
        ptr[offset * 3 + 0] = r;
        ptr[offset * 3 + 1] = g;
        ptr[offset * 3 + 2] = b;
    }
}

void savePPM(const unsigned char *image, const char *filename)
{
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    file << "P6\n"
         << DIM << " " << DIM << "\n255\n";
    for (int i = 0; i < DIM * DIM * 3; i++)
    {
        file << image[i];
    }
    file.close();
}

int main()
{
    unsigned char *h_image = new unsigned char[DIM * DIM * 3];
    unsigned char *d_image;
    cudaMalloc(&d_image, DIM * DIM * 3);

    dim3 blocks(DIM / 32, DIM / 32);
    dim3 threads(32, 32);

    kernel<<<blocks, threads>>>(d_image);

    cudaMemcpy(h_image, d_image, DIM * DIM * 3, cudaMemcpyDeviceToHost);

    savePPM(h_image, "julia.ppm");

    delete[] h_image;
    cudaFree(d_image);

    return 0;
}
