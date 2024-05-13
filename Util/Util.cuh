#include <stdio.h>

namespace info
{
    void cudaInfo()
    {
        cudaDeviceProp prop;
        int device;

        // Get current CUDA device
        cudaGetDevice(&device);

        // Get device properties
        cudaGetDeviceProperties(&prop, device);

        // Print device properties
        printf("Device name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Total global memory: %lu bytes\n", prop.totalGlobalMem);
        printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Maximum threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
        printf("Maximum shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
        printf("Maximum registers per block: %d\n", prop.regsPerBlock);
        printf("Warp size: %d\n", prop.warpSize);
        printf("Memory pitch: %lu bytes\n", prop.memPitch);
        printf("Maximum threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Maximum 1D grid size: %d\n", prop.maxGridSize[0]);
        printf("Maximum 2D grid size: %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1]);
        printf("Maximum 3D grid size: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Maximum 1D block size: %d\n", prop.maxThreadsDim[0]);
        printf("Maximum 2D block size: %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1]);
        printf("Maximum 3D block size: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Clock rate: %d kHz\n", prop.clockRate);
        printf("Texture alignment: %lu bytes\n", prop.textureAlignment);
        printf("Device overlap: %d\n", prop.deviceOverlap);
        printf("Kernel execution timeout enabled: %s\n", (prop.kernelExecTimeoutEnabled ? "Yes" : "No"));
        printf("Integrated GPU sharing host memory: %s\n", (prop.integrated ? "Yes" : "No"));
        printf("Concurrent kernels: %s\n", (prop.concurrentKernels ? "Yes" : "No"));
        printf("Compute mode: %s\n", (prop.computeMode == cudaComputeModeDefault ? "Default" : prop.computeMode == cudaComputeModeExclusive      ? "Exclusive"
                                                                                           : prop.computeMode == cudaComputeModeProhibited       ? "Prohibited"
                                                                                           : prop.computeMode == cudaComputeModeExclusiveProcess ? "Exclusive Process"
                                                                                                                                                 : "Unknown"));
        printf("Number of memory banks: %d\n", prop.memoryBusWidth / 8); // Divide memory bus width by 8 to get number of memory banks
    }
} // namespace info