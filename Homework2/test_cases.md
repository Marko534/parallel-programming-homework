# Test case for example **A**
```bash
5

1 2 3 4 5
6 7 8 9 10
```

# Test case for example **B**
```bash
3 4

1.1 2.2 3.3 4.4
5.5 6.6 7.7 8.8
9.9 10.1 11.11 12.12

0.1 0.2 0.3 0.4
0.5 0.6 0.7 0.8
0.9 1.0 1.1 1.2

```

I used the same device program for adding matracies as for adding vectors. 

Transformed the 2d array in a regular 1d array and then just fed that to the GPU.

After that I just transformed the 1d array back to a 2d array.