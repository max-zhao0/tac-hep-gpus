#include <iostream>

#include <shared.hpp>

#define DSIZE 512
#define STENCIL_RADIUS 2

template <typename T>
T* stencil_2d(const T* M, int width, int radius) {
    T* out = new T[width * width];

    int padded_width = width + 2*radius;
    T* padded = new T[padded_width * padded_width];
    mat::pad(padded, M, width, radius);

    T result;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            result = 0;
            for (int offset = -radius; offset <= radius; offset++) {
                result += padded[i + offset + radius + padded_width * (j + radius)];
                if (offset != 0) {
                    result += padded[i + radius + padded_width * (j + offset + radius)];
                }
            }
            out[i + width * j] = result;
        }
    }

    delete[] padded;
    return out;
}

template <typename T>
T* matrix_mul(const T* A, const T* B, int width) {
    T* out = new T[width * width];

    T temp;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            temp = 0;
            for (int k = 0; k < width; k++){
                temp += A[i * width + k] * B[k * width + j];
            }
            out[i * width + j] = temp;
        }
    }

    return out;
}

// g++ cpp_part.cpp -o cpp.out -I. -g
int main() {
    int* A = new int[DSIZE * DSIZE];
    int* B = new int[DSIZE * DSIZE];
    mat::fill(A, DSIZE, 1);
    mat::fill(B, DSIZE, 1);

    mat::print(A, DSIZE);

    int *A_stenciled = stencil_2d(A, DSIZE, STENCIL_RADIUS);
    int *B_stenciled = stencil_2d(B, DSIZE, STENCIL_RADIUS);

    mat::print(A_stenciled, DSIZE);
    if (!val::stencil_2d(A_stenciled, DSIZE, STENCIL_RADIUS)) return -1;

    int *C = matrix_mul(A_stenciled, B_stenciled, DSIZE);

    mat::print(C, DSIZE);
    if (!val::stenciled_squared(C, DSIZE, STENCIL_RADIUS)) return -1;

    delete[] A;
    delete[] B;
    delete[] A_stenciled;
    delete[] B_stenciled;
    delete[] C;

    std::cout << "Success" << std::endl;
}