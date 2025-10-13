#include <stdio.h>
#include <iostream>


const int DSIZE_X = 256;
const int DSIZE_Y = 256;

__global__ void add_matrix(const float* m1, const float* m2, float* sum, int xdim, int ydim) {
    //FIXME:
    // Express in terms of threads and blocks
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    // Add the two matrices - make sure you are not out of range
    if (idx <  xdim && idy < ydim) {
        int flat_i = idx * xdim + idy;
        sum[flat_i] = m1[flat_i] + m2[flat_i];
    }

}

void print_matrix(const float* mat, int to_print_x=4, int to_print_y=4) {
    std::cout << "[" << std::endl;
    for (int i = 0; i < to_print_x; i++) {
        std::cout << "\t[";
        for (int j = 0; j < to_print_y; j++) {
            std::cout << mat[i * DSIZE_X + j] << ", ";
        }
        std::cout << "... ]," << std::endl;
    }
    std::cout << "\t..." << std::endl;
    std::cout << "]" << std::endl;
}

int main()
{
    // Create and allocate memory for host and device pointers 
    float* M1 = new float[DSIZE_X * DSIZE_Y];
    float* M2 = new float[DSIZE_X * DSIZE_Y];
    float* dM1;
    float* dM2;
    float* dsum;

    int matrix_mem_size = DSIZE_X * DSIZE_Y * sizeof(float);

    cudaMalloc(&dM1, matrix_mem_size);
    cudaMalloc(&dM2, matrix_mem_size);
    cudaMalloc(&dsum, matrix_mem_size);

    // Fill in the matrices
    for (int i = 0; i < DSIZE_X * DSIZE_Y; i++) {
        M1[i] = rand()/(float)RAND_MAX;
        M2[i] = rand()/(float)RAND_MAX;
    }

    // Copy from host to device
    cudaMemcpy(dM1, M1, matrix_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dM2, M2, matrix_mem_size, cudaMemcpyHostToDevice);

    // Launch the kernel
    // dim3 is a built in CUDA type that allows you to define the block 
    // size and grid size in more than 1 dimentions
    // Syntax : dim3(Nx,Ny,Nz)
    int block_width = 16;
    dim3 blockSize(block_width, block_width, 1);
    dim3 gridSize(DSIZE_X / block_width, DSIZE_Y / block_width, 1);
    add_matrix<<<gridSize, blockSize>>>(dM1, dM2, dsum, DSIZE_X, DSIZE_Y);
    cudaDeviceSynchronize();

    // Copy back to host 
    float* sum = new float[DSIZE_X * DSIZE_Y];
    cudaMemcpy(sum, dsum, matrix_mem_size, cudaMemcpyDeviceToHost);

    // Print and check some elements to make the addition was succesfull
    print_matrix(M1);
    std::cout << "\t\t+" << std::endl;
    print_matrix(M2);
    std::cout << "\t\t=" << std::endl;
    print_matrix(sum);

    // Free the memory
    delete[] M1;
    delete[] M2;
    cudaFree(dM1);
    cudaFree(dM2);

    return 0;
}