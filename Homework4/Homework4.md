# Homework 4

## First, run the Python script to generate a test file.

Then, I ran my program.

My conclusion was that it really didn't affect the execution time that much since most of the program's time was spent on writing the data to the GPU and not on the sum calculations. This means that the efficiency of the calculations and threading didn't have that big of an effect. This is what I got for all of the experements with all the diffren homework problems.

Maybe I need to get smaller test samples to see the effect of the threading and block size, since I worked with really big datasets.

To do this, I ran the following command:

```bash
nvprof ./homework4
```
