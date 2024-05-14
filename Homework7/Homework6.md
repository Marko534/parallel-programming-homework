# Homework 7

## C Brent’s theorem

### Definition

Assume a parallel computer where each processor can perform an arithmetic operation in unit time. Further, assume that the computer has exactly enough processors to exploit the maximum concurrency in an algorithm with N operations, such that T time steps suffice. Brent’s Theorem says that a similar computer with fewer processors, P, can perform the algorithm in time

$$ T_P \leq T + \frac{N - T}{P} $$

where \( P \) is less than or equal to the number of processors needed to exploit the maximum concurrency in the algorithm.

### Example

Suppose we have an algorithm that performs $N = 1000$ arithmetic operations and requires $T = 100$ time steps to complete on a parallel computer with enough processors to exploit maximum concurrency.

Let's choose $P = 4$ processors for our example.

Substituting $N = 1000$, $T = 100$, and $P = 4$ into the formula:

$$ T_P \leq 100 + \frac{1000 - 100}{4} $$

$$ T_P \leq 100 + \frac{900}{4} $$

$$ T_P \leq 100 + 225 $$

$$ T_P \leq 325 $$

So, according to Brent's Theorem, if we execute the algorithm on a computer with 4 processors, the time taken will be at most $325$ time steps.

## E

All the programs exept A use serial acumulation for big vectors.

## F

The best improvement is to the local bin histogram compared to the automatic one.
