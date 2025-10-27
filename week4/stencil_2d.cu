#include <stdio.h>
#include <algorithm>
#include <iostream>


using namespace std;

#define N 64
#define RADIUS 2
#define BLOCK_SIZE 32


__global__ void stencil_2d(int *in, int *out) {

	__shared__ int temp[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];
	int gindex_x = threadIdx.x + blockDim.x * blockIdx.x;
	int lindex_x = threadIdx.x + RADIUS;
	int gindex_y = threadIdx.y + blockDim.y * blockIdx.y;
	int lindex_y = threadIdx.y + RADIUS;

	// Read input elements into shared memory
	int size = N + 2 * RADIUS;
	temp[lindex_x][lindex_y] = in[gindex_x + gindex_y * size];

	if (threadIdx.x < RADIUS) {
		temp[lindex_x - RADIUS][lindex_y] = in[gindex_x - RADIUS + gindex_y * size];
		temp[lindex_x + BLOCK_SIZE][lindex_y] = in[gindex_x + BLOCK_SIZE + gindex_y * size];
	}

	if (threadIdx.y < RADIUS ) {
		temp[lindex_x][lindex_y - RADIUS] = in[gindex_x + (gindex_y - RADIUS) * size];
		temp[lindex_x][lindex_y + BLOCK_SIZE] = in[gindex_x + (gindex_y + BLOCK_SIZE) * size];
	}
	__syncthreads();

	// Apply the stencil
	int result = 0;
	for (int offset = -RADIUS; offset <= RADIUS; offset++){
		result += temp[lindex_x + offset][lindex_y];
		if (offset != 0) {
			result += temp[lindex_x][lindex_y + offset];
		}
	}

	// Store the result
	out[gindex_x+size*gindex_y] = result;
}


void fill_ints(int *x, int n) {
   // Store the result
   // https://en.cppreference.com/w/cpp/algorithm/fill_n
   fill_n(x, n, 1);
}

template <typename T>
void print_matrix(const T* mat, int width, int to_print_x=6, int to_print_y=6) {
    std::cout << "[" << std::endl;
    for (int i = 0; i < to_print_x; i++) {
        std::cout << "\t[";
        for (int j = 0; j < to_print_y; j++) {
            std::cout << mat[i * width + j] << ", ";
        }
        std::cout << "... ]," << std::endl;
    }
    std::cout << "\t..." << std::endl;
    std::cout << "]" << std::endl;
}


int main(void) {

	int *in, *out; // host copies of a, b, c
	int *d_in, *d_out; // device copies of a, b, c

	// Alloc space for host copies and setup values
	int size = (N + 2*RADIUS)*(N + 2*RADIUS) * sizeof(int);
	in = (int *)malloc(size); fill_ints(in, (N + 2*RADIUS)*(N + 2*RADIUS));
	out = (int *)malloc(size); fill_ints(out, (N + 2*RADIUS)*(N + 2*RADIUS));

	// Alloc space for device copies
	cudaMalloc((void **)&d_in, size);
	cudaMalloc((void **)&d_out, size);

	// Copy to device
	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

	// Launch stencil_2d() kernel on GPU
	int gridSize = (N + BLOCK_SIZE-1)/BLOCK_SIZE;
	dim3 grid(gridSize, gridSize);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	// Launch the kernel 
	// Properly set memory address for first element on which the stencil will be applied
	stencil_2d<<<grid,block>>>(d_in + RADIUS*(N + 2*RADIUS) + RADIUS , d_out + RADIUS*(N + 2*RADIUS) + RADIUS);
	cudaDeviceSynchronize();

	// Copy result back to host
	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

	// Error Checking
	for (int i = 0; i < N + 2 * RADIUS; ++i) {
		for (int j = 0; j < N + 2 * RADIUS; ++j) {

			if (i < RADIUS || i >= N + RADIUS) {
				if (out[j+i*(N + 2 * RADIUS)] != 1) {
					printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, out[j+i*(N + 2 * RADIUS)], 1);
					return -1;
				}
			}
			else if (j < RADIUS || j >= N + RADIUS) {
				if (out[j+i*(N + 2 * RADIUS)] != 1) {
					printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, out[j+i*(N + 2 * RADIUS)], 1);
					return -1;
				}
			}		 
			else {
				if (out[j+i*(N + 2 * RADIUS)] != 1 + 4 * RADIUS) {
					printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, out[j+i*(N + 2 * RADIUS)], 1 + 4*RADIUS);
					return -1;
				}
			}
		}
	}

	// Cleanup
	free(in);
	free(out);
	cudaFree(d_in);
	cudaFree(d_out);
	printf("Success!\n");

	return 0;
}


