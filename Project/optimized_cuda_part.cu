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

    int lidx = threadIdx.x + radius;
    int lidy = threadIdx.y + radius;
    __shared__ T cache[BLOCK_SIZE + 2*STENCIL_RADIUS][BLOCK_SIZE + 2*STENCIL_RADIUS];

    int pwidth = width + 2*radius;
    int pidx = idx + radius;
    int pidy = idy + radius;

    cache[lidx][lidy] = in_padded[pidx + pwidth * pidy];
    if (threadIdx.x < radius) {
        cache[lidx - radius][lidy] = in_padded[pidx - radius + pwidth * pidy];
        cache[lidx + blockDim.x][lidy] = in_padded[pidx + blockDim.x + pwidth * pidy];
	}

	if (threadIdx.y < radius) {
        cache[lidx][lidy - radius] = in_padded[pidx + pwidth * (pidy - radius)];
        cache[lidx][lidy + blockDim.x] = in_padded[pidx + pwidth * (pidy + blockDim.x)];
	}

    __syncthreads();

    if (idx < width && idy < width) {
        T result = 0;
        for (int offset = -radius; offset <= radius; offset++) {
            result += cache[lidx + offset][lidy];
            if (offset != 0) {
                result += cache[lidx][lidy + offset];
            }
        }
        out[idx + width * idy] = result;
    }
}

template <typename T>
__global__ void matrix_mul(const T* A, const T* B, T* out, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ T cache_A[BLOCK_SIZE][DSIZE];
    // __shared__ T cache_B[DSIZE][BLOCK_SIZE];
    int ca_idx = threadIdx.x;
    // int cb_idy = threadIdx.y;

    for (int ca_idy = threadIdx.y; ca_idy < width; ca_idy += blockDim.y) {
        cache_A[ca_idx][ca_idy] = A[idx + width * ca_idy];
    }
    // for (int cb_idx = threadIdx.x; cb_idx < width; cb_idx += blockDim.x) {
    //     cache_B[cb_idx][cb_idy] = B[cb_idx + width * idy];
    // }

    __syncthreads();
    
    if (idx < width && idy < width) {
        T result = 0;
        for (int k = 0; k < width; k++){
            // result += cache_A[ca_idx][k] * cache_B[k][cb_idy];
            result += cache_A[ca_idx][k] * B[k + width * idy];
        }
        out[idx * width + idy] = result;
    }
}

// nvcc optimized_cuda_part.cu -o optimized_cuda.out -I.
int main() {
    int* A = new int[DSIZE * DSIZE];
    int* B = new int[DSIZE * DSIZE];
    mat::fill(A, DSIZE, 1);
    mat::fill(B, DSIZE, 1);
    mat::print(A, DSIZE);

    int padded_width = DSIZE + 2*STENCIL_RADIUS;
    int padded_mem_size = padded_width * padded_width * sizeof(int);
    int *A_padded, *B_padded;
    cudaMallocHost(&A_padded, padded_mem_size);
    cudaMallocHost(&B_padded, padded_mem_size);
    mat::pad(A_padded, A, DSIZE, STENCIL_RADIUS);
    mat::pad(B_padded, B, DSIZE, STENCIL_RADIUS);
    
    int mem_size = DSIZE * DSIZE * sizeof(int);
    int *d_A_padded, *d_B_padded, *d_A_stenciled, *d_B_stenciled;

    cudaMalloc(&d_A_padded, padded_mem_size);
    cudaMalloc(&d_B_padded, padded_mem_size);
    cudaMalloc(&d_A_stenciled, mem_size);
    cudaMalloc(&d_B_stenciled, mem_size);
    cudaCheckErrors("Error while allocating padded and stenciled");

    cudaStream_t stream_A, stream_B;
    cudaStreamCreate(&stream_A);
    cudaStreamCreate(&stream_B);
    cudaCheckErrors("Error while creating streams");

    cudaMemcpyAsync(d_A_padded, A_padded, padded_mem_size, cudaMemcpyHostToDevice, stream_A);
    cudaMemcpyAsync(d_B_padded, B_padded, padded_mem_size, cudaMemcpyHostToDevice, stream_B);
    cudaCheckErrors("Error while copying padded to device");

    int grid_size = (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block(BLOCK_SIZE,BLOCK_SIZE,1);
    dim3 grid(grid_size, grid_size, 1);

    stencil_2d<<<grid, block, 0, stream_A>>>(d_A_padded, d_A_stenciled, DSIZE, STENCIL_RADIUS);
    stencil_2d<<<grid, block, 0, stream_B>>>(d_B_padded, d_B_stenciled, DSIZE, STENCIL_RADIUS);
    cudaCheckErrors("Error during stencil_2d");

    int* A_stenciled;
    cudaMallocHost(&A_stenciled, mem_size);
    cudaMemcpyAsync(A_stenciled, d_A_stenciled, mem_size, cudaMemcpyDeviceToHost, stream_A);
    cudaCheckErrors("Error while copying stencil to host");

    int *d_C;
    cudaMalloc(&d_C, mem_size);
    cudaCheckErrors("Error while allocating C memory");

    matrix_mul<<<grid, block, 0, stream_B>>>(d_A_stenciled, d_B_stenciled, d_C, DSIZE);
    cudaCheckErrors("Error during matmul");

    int* C;
    cudaMallocHost(&C, mem_size);
    cudaMemcpyAsync(C, d_C, mem_size, cudaMemcpyDeviceToHost, stream_B);
    cudaCheckErrors("Error while copying C to host");

    cudaStreamSynchronize(stream_A);
    mat::print(A_stenciled, DSIZE);
    if (!val::stencil_2d(A_stenciled, DSIZE, STENCIL_RADIUS)) return -1;

    cudaStreamSynchronize(stream_B);
    mat::print(C, DSIZE);
    if (!val::stenciled_squared(C, DSIZE, STENCIL_RADIUS)) return -1;

    cudaStreamDestroy(stream_A);
    cudaStreamDestroy(stream_B);
    
    delete[] A;
    delete[] B;
    cudaFreeHost(A_padded);
    cudaFreeHost(B_padded);
    cudaFreeHost(A_stenciled);
    cudaFreeHost(C);
    cudaFree(d_A_padded);
    cudaFree(d_B_padded);
    cudaFree(d_A_stenciled);
    cudaFree(d_B_stenciled);
    cudaFree(d_C);

    std::cout << "Success" << std::endl;
}