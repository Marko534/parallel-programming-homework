# Homework 5

## A

### Timing

<!-- This is a table -->

| Method               | Time (µs) |
| -------------------- | --------- |
| CPU                  | 32.895    |
| GPU serial           | 51.072    |
| GPU row parallel     | 8.2880    |
| GPU element parallel | 2.7840    |

To get thise results I used this bash comand.

```bash
nvprof ./homework5
```

We can see that one thread of the CPU is faster than the GPU serial version. The GPU row parallel is faster than the CPU because it is working in parallel so the slower thread speed of the GPU is negated. The same story is for GPU element parallel, here instead of therad for every row we have one for every element.

## B

This is how the kernal could be made to run in parallel.
I have only 3 colors since the image format I am using only uses 3 colors.

```c++
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
            return 0;
    }
    return 1;
}

__global__ void kernel(unsigned char *ptr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < DIM && y < DIM)
    {
        int offset = x + y * DIM;
        int juliaValue = julia(x, y);
        ptr[offset * 3 + 0] = 255 * juliaValue;
        ptr[offset * 3 + 1] = 0;
        ptr[offset * 3 + 2] = 0;
    }
}

```

This is the result of the kernal running in parallel.
![Julia Set](b/julia.png)
I used a different kernal for this one to get a better image. The source code is in the homework5.cu file.

![Julia Set](<b/julia(1).png>)

I used the ppm format because it is easy to write to fiele. I transformed the images to png so they take up less space.

## C

Used some code from the problem where we had to find the max of a array.
