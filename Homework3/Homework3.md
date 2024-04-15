# Homework3

## First run the python script to generate a test file

The I ran my program for adding matricies together. I changed the number of threads and the number of blocks too.

My conclusion was that it really didn't effect the execution time that much since most of the programs time was spent on writing the data to the gpu and not the sum calculations. This means that the eficincy of the calculations and threading didn't have that big of an effect.

To do this I ran the following comand.

```bash
nvprof ./homework3
```

I got the following output for 8 blocks each block with 1024 threads adding two matricies with size 8192 x 8192.

```bash

==107271== NVPROF is profiling process 107271, command: ./homework3
==107271== Profiling application: ./homework3
==107271== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   62.44%  150.21ms         1  150.21ms  150.21ms  150.21ms  [CUDA memcpy DtoH]
                   37.56%  90.366ms         2  45.183ms  44.704ms  45.662ms  [CUDA memcpy HtoD]
                    0.00%  2.5920us         1  2.5920us  2.5920us  2.5920us  matrix_add(float const *, float const *, float*)
      API calls:   73.14%  241.71ms         3  80.569ms  44.795ms  151.21ms  cudaMemcpy
                   26.42%  87.326ms         3  29.109ms  79.759us  87.135ms  cudaMalloc
                    0.26%  874.35us         3  291.45us  260.86us  348.65us  cudaFree
                    0.09%  306.12us       114  2.6850us     907ns  70.959us  cuDeviceGetAttribute
                    0.06%  209.87us         1  209.87us  209.87us  209.87us  cudaLaunchKernel
                    0.01%  21.023us         1  21.023us  21.023us  21.023us  cuDeviceGetName
                    0.00%  10.476us         1  10.476us  10.476us  10.476us  cuDeviceGetPCIBusId
                    0.00%  5.0990us         3  1.6990us  1.3970us  2.3050us  cuDeviceGetCount
                    0.00%  2.4440us         2  1.2220us     977ns  1.4670us  cuDeviceGet
                    0.00%  1.4660us         1  1.4660us  1.4660us  1.4660us  cuDeviceGetUuid
                    0.00%  1.3970us         1  1.3970us  1.3970us  1.3970us  cuModuleGetLoadingMode
                    0.00%     978ns         1     978ns     978ns     978ns  cuDeviceTotalMem ~14s

```
