# Homework 6

## A

### Tile Size 8

<!-- This is a table -->

| Type           | Time(%) | Time     | Calls | Avg      | Min      | Max      | Name                                        |
| -------------- | ------- | -------- | ----- | -------- | -------- | -------- | ------------------------------------------- |
| GPU activities | 55.41%  | 15.783ms | 1     | 15.783ms | 15.783ms | 15.783ms | basic_mat_mul(float*, float*, float\*, int) |
|                | 35.01%  | 9.9729ms | 1     | 9.9729ms | 9.9729ms | 9.9729ms | fast_mat_mul(float*, float*, float\*, int)  |
|                | 5.11%   | 1.4544ms | 1     | 1.4544ms | 1.4544ms | 1.4544ms | [CUDA memcpy DtoH]                          |
|                | 4.47%   | 1.2735ms | 2     | 636.77us | 636.32us | 637.21us | [CUDA memcpy HtoD]                          |
| API calls      | 71.97%  | 78.045ms | 3     | 26.015ms | 66.140us | 77.872ms | cudaMalloc                                  |
|                | 27.21%  | 29.501ms | 3     | 9.8336ms | 622.50us | 28.175ms | cudaMemcpy                                  |
|                | 0.28%   | 307.01us | 114   | 2.6930us | 907ns    | 73.473us | cuDeviceGetAttribute                        |
|                | 0.25%   | 273.57us | 3     | 91.190us | 55.385us | 150.44us | cudaFree                                    |
|                | 0.24%   | 260.02us | 2     | 130.01us | 6.2160us | 253.80us | cudaLaunchKernel                            |
|                | 0.02%   | 21.930us | 1     | 21.930us | 21.930us | 21.930us | cuDeviceGetName                             |
|                | 0.02%   | 17.041us | 1     | 17.041us | 17.041us | 17.041us | cuDeviceGetPCIBusId                         |
|                | 0.00%   | 4.1910us | 3     | 1.3970us | 1.3970us | 1.3970us | cuDeviceGetCount                            |
|                | 0.00%   | 2.7930us | 2     | 1.3960us | 1.3960us | 1.3970us | cuDeviceGet                                 |
|                | 0.00%   | 1.8860us | 1     | 1.8860us | 1.8860us | 1.8860us | cuDeviceGetUuid                             |
|                | 0.00%   | 1.4660us | 1     | 1.4660us | 1.4660us | 1.4660us | cuModuleGetLoadingMode                      |
|                | 0.00%   | 1.4660us | 1     | 1.4660us | 1.4660us | 1.4660us | cuDeviceTotalMem                            |

### Tile Size 16

| Type           | Time(%) | Time     | Calls | Avg      | Min      | Max      | Name                                        |
| -------------- | ------- | -------- | ----- | -------- | -------- | -------- | ------------------------------------------- |
| GPU activities | 53.08%  | 10.259ms | 1     | 10.259ms | 10.259ms | 10.259ms | basic_mat_mul(float*, float*, float\*, int) |
|                | 32.76%  | 6.3324ms | 1     | 6.3324ms | 6.3324ms | 6.3324ms | fast_mat_mul(float*, float*, float\*, int)  |
|                | 7.57%   | 1.4638ms | 1     | 1.4638ms | 1.4638ms | 1.4638ms | [CUDA memcpy DtoH]                          |
|                | 6.59%   | 1.2737ms | 2     | 636.86us | 636.77us | 636.96us | [CUDA memcpy HtoD]                          |
| API calls      | 78.82%  | 79.183ms | 3     | 26.394ms | 68.654us | 79.004ms | cudaMalloc                                  |
|                | 20.27%  | 20.367ms | 3     | 6.7889ms | 618.38us | 19.026ms | cudaMemcpy                                  |
|                | 0.32%   | 324.97us | 114   | 2.8500us | 907ns    | 76.266us | cuDeviceGetAttribute                        |
|                | 0.27%   | 272.94us | 3     | 90.980us | 56.781us | 149.39us | cudaFree                                    |
|                | 0.26%   | 263.79us | 2     | 131.90us | 6.2160us | 257.58us | cudaLaunchKernel                            |
|                | 0.02%   | 24.235us | 1     | 24.235us | 24.235us | 24.235us | cuDeviceGetName                             |
|                | 0.01%   | 9.4990us | 1     | 9.4990us | 9.4990us | 9.4990us | cuDeviceGetPCIBusId                         |
|                | 0.00%   | 4.8200us | 3     | 1.6060us | 978ns    | 2.4450us | cuDeviceGetCount                            |
|                | 0.00%   | 3.3520us | 2     | 1.6760us | 1.4660us | 1.8860us | cuDeviceGet                                 |
|                | 0.00%   | 1.4660us | 1     | 1.4660us | 1.4660us | 1.4660us | cuDeviceTotalMem                            |
|                | 0.00%   | 1.3970us | 1     | 1.3970us | 1.3970us | 1.3970us | cuModuleGetLoadingMode                      |
|                | 0.00%   | 1.3970us | 1     | 1.3970us | 1.3970us | 1.3970us | cuDeviceGetUuid                             |

### Tile Size 32

| Type           | Time(%) | Time     | Calls | Avg      | Min      | Max      | Name                                        |
| -------------- | ------- | -------- | ----- | -------- | -------- | -------- | ------------------------------------------- |
| GPU activities | 46.21%  | 7.4141ms | 1     | 7.4141ms | 7.4141ms | 7.4141ms | basic_mat_mul(float*, float*, float\*, int) |
|                | 36.49%  | 5.8553ms | 1     | 5.8553ms | 5.8553ms | 5.8553ms | fast_mat_mul(float*, float*, float\*, int)  |
|                | 9.36%   | 1.5014ms | 1     | 1.5014ms | 1.5014ms | 1.5014ms | [CUDA memcpy DtoH]                          |
|                | 7.95%   | 1.2751ms | 2     | 637.53us | 636.92us | 638.14us | [CUDA memcpy HtoD]                          |
| API calls      | 81.70%  | 80.282ms | 3     | 26.761ms | 67.397us | 80.114ms | cudaMalloc                                  |
|                | 17.38%  | 17.081ms | 3     | 5.6935ms | 628.29us | 15.749ms | cudaMemcpy                                  |
|                | 0.32%   | 314.08us | 114   | 2.7550us | 907ns    | 73.334us | cuDeviceGetAttribute                        |
|                | 0.28%   | 278.04us | 3     | 92.679us | 55.244us | 156.17us | cudaFree                                    |
|                | 0.26%   | 256.25us | 2     | 128.12us | 6.1460us | 250.10us | cudaLaunchKernel                            |
|                | 0.02%   | 20.812us | 1     | 20.812us | 20.812us | 20.812us | cuDeviceGetName                             |
|                | 0.02%   | 16.063us | 1     | 16.063us | 16.063us | 16.063us | cuDeviceGetPCIBusId                         |
|                | 0.00%   | 4.7490us | 3     | 1.5830us | 1.3970us | 1.8860us | cuDeviceGetCount                            |
|                | 0.00%   | 2.7940us | 2     | 1.3970us | 908ns    | 1.8860us | cuDeviceGet                                 |
|                | 0.00%   | 1.4670us | 1     | 1.4670us | 1.4670us | 1.4670us | cuModuleGetLoadingMode                      |
|                | 0.00%   | 1.4670us | 1     | 1.4670us | 1.4670us | 1.4670us | cuDeviceTotalMem                            |
|                | 0.00%   | 978ns    | 1     | 978ns    | 978ns    | 978ns    | cuDeviceGetUuid                             |

To get thise results I used this bash comand.

```bash
nvprof ./homework6
```

The tile size of 32 is the fastest. The tile size of 16 is the second fastest. The tile size of 8 is the slowest. The preformance improvments drop off the closer we get to the tile size of 32. For my GPU that is the max since 32\*32 = 1024 which is the max number of threads per block. Even when using 8 tiles the preformance is still better than the basic matrix multiplication.

## B

### Tile Size 8

| Type           | Time(%) | Time     | Calls | Avg      | Min      | Max      | Name                                      |
| -------------- | ------- | -------- | ----- | -------- | -------- | -------- | ----------------------------------------- |
| GPU activities | 66.10%  | 1.5181ms | 1     | 1.5181ms | 1.5181ms | 1.5181ms | [CUDA memcpy DtoH]                        |
|                | 27.98%  | 642.72us | 1     | 642.72us | 642.72us | 642.72us | [CUDA memcpy HtoD]                        |
|                | 3.08%   | 70.688us | 1     | 70.688us | 70.688us | 70.688us | fast_transpose(float*, float*, int, int)  |
|                | 2.84%   | 65.281us | 1     | 65.281us | 65.281us | 65.281us | basic_transpose(float*, float*, int, int) |
| API calls      | 94.92%  | 76.641ms | 2     | 38.321ms | 107.21us | 76.534ms | cudaMalloc                                |
|                | 4.07%   | 3.2888ms | 2     | 1.6444ms | 605.04us | 2.6837ms | cudaMemcpy                                |
|                | 0.40%   | 324.00us | 114   | 2.8420us | 908ns    | 75.988us | cuDeviceGetAttribute                      |
|                | 0.29%   | 233.27us | 2     | 116.64us | 6.1460us | 227.13us | cudaLaunchKernel                          |
|                | 0.25%   | 202.40us | 2     | 101.20us | 64.883us | 137.52us | cudaFree                                  |
|                | 0.03%   | 23.886us | 1     | 23.886us | 23.886us | 23.886us | cuDeviceGetName                           |
|                | 0.02%   | 19.556us | 1     | 19.556us | 19.556us | 19.556us | cuDeviceGetPCIBusId                       |
|                | 0.01%   | 4.6790us | 3     | 1.5590us | 1.3960us | 1.8860us | cuDeviceGetCount                          |
|                | 0.00%   | 3.2820us | 2     | 1.6410us | 1.3970us | 1.8850us | cuDeviceGet                               |
|                | 0.00%   | 1.3970us | 1     | 1.3970us | 1.3970us | 1.3970us | cuDeviceTotalMem                          |
|                | 0.00%   | 1.3960us | 1     | 1.3960us | 1.3960us | 1.3960us | cuDeviceGetUuid                           |
|                | 0.00%   | 977ns    | 1     | 977ns    | 977ns    | 977ns    | cuModuleGetLoadingMode                    |

### Tile Size 16

| Type           | Time(%) | Time     | Calls | Avg      | Min      | Max      | Name                                      |
| -------------- | ------- | -------- | ----- | -------- | -------- | -------- | ----------------------------------------- |
| GPU activities | 66.18%  | 1.5402ms | 1     | 1.5402ms | 1.5402ms | 1.5402ms | [CUDA memcpy DtoH]                        |
|                | 27.66%  | 643.64us | 1     | 643.64us | 643.64us | 643.64us | [CUDA memcpy HtoD]                        |
|                | 3.19%   | 74.239us | 1     | 74.239us | 74.239us | 74.239us | basic_transpose(float*, float*, int, int) |
|                | 2.97%   | 69.056us | 1     | 69.056us | 69.056us | 69.056us | fast_transpose(float*, float*, int, int)  |
| API calls      | 95.17%  | 86.324ms | 2     | 43.162ms | 111.33us | 86.213ms | cudaMalloc                                |
|                | 3.66%   | 3.3160ms | 2     | 1.6580ms | 638.98us | 2.6770ms | cudaMemcpy                                |
|                | 0.58%   | 523.95us | 114   | 4.5960us | 1.3960us | 132.70us | cuDeviceGetAttribute                      |
|                | 0.27%   | 248.50us | 2     | 124.25us | 72.006us | 176.49us | cudaFree                                  |
|                | 0.26%   | 236.76us | 2     | 118.38us | 27.168us | 209.59us | cudaLaunchKernel                          |
|                | 0.03%   | 28.565us | 1     | 28.565us | 28.565us | 28.565us | cuDeviceGetName                           |
|                | 0.01%   | 9.9170us | 1     | 9.9170us | 9.9170us | 9.9170us | cuDeviceGetPCIBusId                       |
|                | 0.01%   | 7.1930us | 3     | 2.3970us | 1.8860us | 2.9330us | cuDeviceGetCount                          |
|                | 0.01%   | 4.7490us | 2     | 2.3740us | 1.9550us | 2.7940us | cuDeviceGet                               |
|                | 0.00%   | 2.4450us | 1     | 2.4450us | 2.4450us | 2.4450us | cuModuleGetLoadingMode                    |
|                | 0.00%   | 1.8860us | 1     | 1.8860us | 1.8860us | 1.8860us | cuDeviceTotalMem                          |
|                | 0.00%   | 1.8860us | 1     | 1.8860us | 1.8860us | 1.8860us | cuDeviceGetUuid                           |

### Tile Size 32

| Type           | Time(%) | Time     | Calls | Avg      | Min      | Max      | Name                                      |
| -------------- | ------- | -------- | ----- | -------- | -------- | -------- | ----------------------------------------- |
| GPU activities | 63.63%  | 1.5184ms | 1     | 1.5184ms | 1.5184ms | 1.5184ms | [CUDA memcpy DtoH]                        |
|                | 26.75%  | 638.49us | 1     | 638.49us | 638.49us | 638.49us | [CUDA memcpy HtoD]                        |
|                | 5.96%   | 142.34us | 1     | 142.34us | 142.34us | 142.34us | basic_transpose(float*, float*, int, int) |
|                | 3.66%   | 87.263us | 1     | 87.263us | 87.263us | 87.263us | fast_transpose(float*, float*, int, int)  |
| API calls      | 94.63%  | 75.673ms | 2     | 37.836ms | 136.68us | 75.536ms | cudaMalloc                                |
|                | 4.35%   | 3.4752ms | 2     | 1.7376ms | 630.18us | 2.8451ms | cudaMemcpy                                |
|                | 0.41%   | 325.88us | 114   | 2.8580us | 907ns    | 88.209us | cuDeviceGetAttribute                      |
|                | 0.29%   | 235.86us | 2     | 117.93us | 6.2860us | 229.57us | cudaLaunchKernel                          |
|                | 0.26%   | 209.80us | 2     | 104.90us | 72.355us | 137.45us | cudaFree                                  |
|                | 0.03%   | 22.000us | 1     | 22.000us | 22.000us | 22.000us | cuDeviceGetName                           |
|                | 0.01%   | 9.4280us | 1     | 9.4280us | 9.4280us | 9.4280us | cuDeviceGetPCIBusId                       |
|                | 0.01%   | 4.7500us | 3     | 1.5830us | 1.3970us | 1.9560us | cuDeviceGetCount                          |
|                | 0.00%   | 3.3520us | 2     | 1.6760us | 1.4660us | 1.8860us | cuDeviceGet                               |
|                | 0.00%   | 1.8160us | 1     | 1.8160us | 1.8160us | 1.8160us | cuDeviceTotalMem                          |
|                | 0.00%   | 1.4670us | 1     | 1.4670us | 1.4670us | 1.4670us | cuModuleGetLoadingMode                    |
|                | 0.00%   | 1.3970us | 1     | 1.3970us | 1.3970us | 1.3970us | cuDeviceGetUuid                           |

To get thise results I used this bash comand.

```bash
nvprof ./homework6
```

The results for the transpose are simmilar as well as the impruvments with using more tiles.

## C

I didn't understand the task.
