#include <stdio.h>
#include <time.h>

const int DSIZE = 256;
const float A_val = 3.0f;
const float B_val = 2.0f;

// error checking macro
#define cudaCheckErrors(msg)                                   \
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

// Square matrix multiplication on CPU : C = A * B
void matrix_mul_cpu(const float *A, const float *B, float *C, int size) {
    for (int idx = 0; idx < size; idx++) {
        for (int idy = 0; idy < size; idy++) {
            float temp = 0;
            for (int i = 0; i < size; i++){
                temp += A[idx * size + i] * B[i * size + idy];
            }
            C[idx * size + idy] = temp;
        }
    }
}

// Square matrix multiplication on GPU : C = A * B
__global__ void matrix_mul_gpu(const float *A, const float *B, float *C, int size) {
    // create thread x index
    // create thread y index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    // Make sure we are not out of range
    if ((idx < size) && (idy < size)) {
        float temp = 0;
        for (int i = 0; i < size; i++){
            temp += A[idx * size + i] * B[i * size + idy];
        }
        C[idx * size + idy] = temp;                    
    }
}

int main() {

    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    // These are used for timing
    clock_t t0, t1, t2, t3;
    double t1sum=0.0;
    double t2sum=0.0;
    double t3sum=0.0;

    // start timing
    t0 = clock();

    // N*N matrices defined in 1 dimention
    // If you prefer to do this in 2-dimentions cupdate accordingly
    h_A = new float[DSIZE*DSIZE];
    h_B = new float[DSIZE*DSIZE];
    h_C = new float[DSIZE*DSIZE];
    for (int i = 0; i < DSIZE*DSIZE; i++){
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C[i] = 0;
    }

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    // Allocate device memory and copy input data from host to device
    int array_mem_size = DSIZE*DSIZE*sizeof(float);

    cudaMalloc(&d_A, array_mem_size);
    cudaMalloc(&d_B, array_mem_size);
    cudaMalloc(&d_C, array_mem_size);
    cudaCheckErrors("Error while allocating device memory");

    cudaMemcpy(d_A, h_A, array_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, array_mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("Error while copying memory to device");

    // Launch kernel
    // Specify the block and grid dimentions
    int block_width = 16;
    dim3 block(block_width,block_width,1);  //FIXME
    dim3 grid(DSIZE / block_width,DSIZE / block_width,1); //FIXME
    matrix_mul_gpu<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_C, d_C, array_mem_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("Error while copying memory to host");

    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
    printf ("Done. Compute took %f seconds\n", t2sum);

    // FIXME
    // Excecute and time the cpu matrix multiplication function
    matrix_mul_cpu(h_A, h_B, h_C, DSIZE);

    // CPU timing
    t3 = clock();
    t3sum = ((double)(t3-t2))/CLOCKS_PER_SEC;
    printf ("Done. Compute took %f seconds\n", t3sum);

    // Free memory 
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;

}
