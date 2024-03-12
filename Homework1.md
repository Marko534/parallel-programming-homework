# Homework 1
1. **Latency** helps (or hurts) **bandwidth**:
	- **Latency** and **bandwidth** are not correlated. 
		-  **Latency** is the delay from the start of the calculation to getting a result, it is measured time delay.
		- **Bandwidth** is the amount of parallel calculation that can accrue.
	- Usually it is a trade-off between the two and in parallel programming we focus on increasing the the **bandwidth** and we don't care that much about the **latency** because the gains that we can get from **decreasing** it is negligible compared to the benefit we get from increasing the **bandwidth**.
	- But by the restrictions of the hardware we use (CPU vs GPU) there is a trade-off.
		- **CPUs** have low **latency** (which is good) and low **bandwidth** (which is bad).
  		- **GPUs** have higher **latency** then **GPUs** and a way higher **bandwidth**.
	- So by increasing the **latency** (by using a **GPU**) we are increasing the **bandwidth**. And the opposite is true if we are using a **CPU**
2. **Bandwidth** helps (or hurts) **latency**:
	- (see answer for question 1)
	- So by increasing the **bandwidth** (by using a **GPU**) we are increasing the **latency**. And the opposite is true if we are using a **CPU**
3. Software overhead helps (or hurts) **latency**:
	- Overhead hurts the **latency** because there are more instructions the processor has to do thus lowering the response time.
4. **Latency** (or **bandwidth**) is easier to sell:
	- Depends on the thing that you are trying to sell:
		- Low **latency** is easier to sell if you need fast response time and only need to be able to do calculations in serial ( think networking, impeded systems, complex control flow etc.)
		- High **bandwidth** if you need to do a LOT of simple calculations ( think computer graphics, machine learning etc.)
5. Caching helps (or hurts) **latency**:
	- It help **latency** because the processor can use the cashed date in the fast memory and doesn't need to waist time reading something from the main memory (which is comparatively slower)
6. Replication: leverages capacity to hurt (or
help) **latency** :
	- If we intend to do the same calculation over and over it makes sense to do them in parallel. This entails that we make the **latency** higher for the individual calculations **BUT** the whole program will finish executing faster since we are doing the calculations in parallel.
7. Prediction: leverages **bandwidth** to help
(or hurt) **latency**:
	- I don't know what is meant here by prediction.

































