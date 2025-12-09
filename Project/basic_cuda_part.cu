#include <iostream>

#include <shared.hpp>

#define DSIZE 512
#define STENCIL_RADIUS 2
#define BLOCK_SIZE 16

#define cudaCheckErrors(msg)                                    \
    do {                                                        \
        cudaError_t __err = cudaGetLastError();                 \
        if (__err != cudaSuccess) {                             \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                    msg, cudaGetErrorString(__err),             \
                    __FILE__, __LINE__);                        \
            fprintf(stderr, "*** FAILED - ABORTING\n");         \
            exit(1);                                            \
        }                                                       \
    } while (0)

template <typename T>
__global__ void stencil_2d(const T* in_padded, T* out, int width, int radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < width) {
        int pwidth = width + 2*radius;
        int pidx = idx + radius;
        int pidy = idy + radius;
        T result = 0;
        for (int offset = -radius; offset <= radius; offset++) {
            result += in_padded[pidx + offset + pwidth * pidy];
            if (offset != 0) {
                result += in_padded[pidx + pwidth * (pidy + offset)];
            }
        }
        out[idx + width * idy] = result;
    }
}

template <typename T>
__global__ void matrix_mul(const T* A, const T* B, T* out, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < width) {
        T result = 0;
        for (int k = 0; k < width; k++){
            result += A[idx * width + k] * B[k * width + idy];
        }
        out[idx * width + idy] = result;
    }
}

// nvcc basic_cuda_part.cu -o basic_cuda.out -I.
int main() {
    int* A = new int[DSIZE * DSIZE];
    int* B = new int[DSIZE * DSIZE];
    mat::fill(A, DSIZE, 1);
    mat::fill(B, DSIZE, 1);
    mat::print(A, DSIZE);

    int padded_width = DSIZE + 2*STENCIL_RADIUS;
    int* A_padded = new int[padded_width * padded_width];
    int* B_padded = new int[padded_width * padded_width];
    mat::pad(A_padded, A, DSIZE, STENCIL_RADIUS);
    mat::pad(B_padded, B, DSIZE, STENCIL_RADIUS);
    
    int padded_mem_size = padded_width * padded_width * sizeof(int);
    int mem_size = DSIZE * DSIZE * sizeof(int);
    int *d_A_padded, *d_B_padded, *d_A_stenciled, *d_B_stenciled;

    cudaMalloc(&d_A_padded, padded_mem_size);
    cudaMalloc(&d_B_padded, padded_mem_size);
    cudaMalloc(&d_A_stenciled, mem_size);
    cudaMalloc(&d_B_stenciled, mem_size);
    cudaCheckErrors("Error while allocating padded and stenciled");

    cudaMemcpy(d_A_padded, A_padded, padded_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_padded, B_padded, padded_mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("Error while copying padded to device");

    int grid_size = (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block(BLOCK_SIZE,BLOCK_SIZE,1);
    dim3 grid(grid_size, grid_size, 1);

    stencil_2d<<<grid, block>>>(d_A_padded, d_A_stenciled, DSIZE, STENCIL_RADIUS);
    cudaDeviceSynchronize();
    stencil_2d<<<grid, block>>>(d_B_padded, d_B_stenciled, DSIZE, STENCIL_RADIUS);
    cudaDeviceSynchronize();
    cudaCheckErrors("Error during stencil_2d");

    int* A_stenciled = new int[DSIZE*DSIZE];
    cudaMemcpy(A_stenciled, d_A_stenciled, mem_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("Error while copying stencil to host");
    mat::print(A_stenciled, DSIZE);
    if (!val::stencil_2d(A_stenciled, DSIZE, STENCIL_RADIUS)) return -1;

    int *d_C;
    cudaMalloc(&d_C, mem_size);
    cudaCheckErrors("Error while allocating C memory");

    matrix_mul<<<grid, block>>>(d_A_stenciled, d_B_stenciled, d_C, DSIZE);
    cudaDeviceSynchronize();
    cudaCheckErrors("Error during matmul");

    int* C = new int[DSIZE * DSIZE];
    cudaMemcpy(C, d_C, mem_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("Error while copying C to host");
    mat::print(C, DSIZE);
    if (!val::stenciled_squared(C, DSIZE, STENCIL_RADIUS)) return -1;
    
    delete[] A;
    delete[] B;
    delete[] A_padded;
    delete[] B_padded;
    delete[] A_stenciled;
    delete[] C;
    cudaFree(d_A_padded);
    cudaFree(d_B_padded);
    cudaFree(d_A_stenciled);
    cudaFree(d_B_stenciled);
    cudaFree(d_C);

    std::cout << "Success" << std::endl;
}