#include <stdio.h>
#include <time.h>
#include <iostream>

#define BLOCK_SIZE 32

const int DSIZE = 256;
const int a = 1;
const int b = 1;

// error checking macro
#define cudaCheckErrors()                                       \
	do {                                                        \
		cudaError_t __err = cudaGetLastError();                 \
		if (__err != cudaSuccess) {                             \
			fprintf(stderr, "Error:  %s at %s:%d \n",           \
			cudaGetErrorString(__err),__FILE__, __LINE__);      \
			fprintf(stderr, "*** FAILED - ABORTING***\n");      \
			exit(1);                                            \
		}                                                       \
	} while (0)


// CUDA kernel that runs on the GPU
__global__ void dot_product(const int *A, const int *B, int *C, int N) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < N) {
		atomicAdd(C, A[idx] * B[idx]);
	}
}

template <typename T>
void print_vector(T* vec, int to_print=20) {
    std::cout << "[";
    for (int i = 0; i < to_print; i++) {
        std::cout << vec[i] << ", ";
    }
    std::cout << "... ]" << std::endl;
}

int main() {
	
	// Create the device and host pointers
	int *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

	// Fill in the host pointers 
	h_A = new int[DSIZE];
	h_B = new int[DSIZE];
	h_C = new int;
	for (int i = 0; i < DSIZE; i++){
		h_A[i] = a;
		h_B[i] = b;
	}

	*h_C = 0;

	// Allocate device memory
	int vec_mem_size = DSIZE * sizeof(int);
	cudaMalloc(&d_A, vec_mem_size);
	cudaMalloc(&d_B, vec_mem_size);
	cudaMalloc(&d_C, sizeof(int));
	cudaCheckErrors(); 

	// Copy the matrices on GPU
	cudaMemcpy(d_A, h_A, vec_mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, vec_mem_size, cudaMemcpyHostToDevice);
	cudaCheckErrors();

	// Define block/grid dimentions and launch kernel
	int block_size = 256;
	int grid_size = DSIZE / block_size;
	dot_product<<<grid_size,block_size>>>(d_A, d_B, d_C, DSIZE);
	cudaDeviceSynchronize();
	cudaCheckErrors();
	
	// Copy results back to host
	cudaMemcpy(h_C, d_C, sizeof(int), cudaMemcpyDeviceToHost);
	cudaCheckErrors();

	// Verify result
	print_vector(h_A);
	std::cout << "\t\t•" << std::endl;
	print_vector(h_B);
	std::cout << "\t\t= " << *h_C << std::endl;

	// Free allocated memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaCheckErrors();

	delete[] h_A;
	delete[] h_B;
	delete h_C;
	
	return 0;

}