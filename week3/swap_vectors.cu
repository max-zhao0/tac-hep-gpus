#include <stdio.h>
#include <iostream>


const int DSIZE = 40960;
const int block_size = 256;
const int grid_size = DSIZE/block_size;


__global__ void vector_swap(float* vec1, float* vec2, float* buffer, int size) {
    // Express the vector index in terms of threads and blocks
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Swap the vector elements - make sure you are not out of range
    if (idx < size) {
        buffer[idx] = vec1[idx];
        vec1[idx] = vec2[idx];
        vec2[idx] = buffer[idx];
    }
}

void print_vector(float* vec, int to_print=20) {
    std::cout << "[";
    for (int i = 0; i < to_print; i++) {
        std::cout << vec[i] << ", ";
    }
    std::cout << "... ]" << std::endl;
}

int main() {
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];


    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
        h_C[i] = 0;
    }

    print_vector(h_A);
    print_vector(h_B);

    // Allocate memory for host and device pointers 
    cudaMalloc(&d_A, DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * sizeof(float));
    cudaMalloc(&d_C, DSIZE * sizeof(float));

    // Copy from host to device
    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    vector_swap<<<grid_size,block_size>>>(d_A, d_B, d_C, DSIZE);
    cudaDeviceSynchronize();

    // Copy back to host 
    cudaMemcpy(h_A, d_A, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print and check some elements to make sure swapping was successfull
    std::cout << std::endl;
    print_vector(h_A);
    print_vector(h_B);

    // Free the memory 
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
