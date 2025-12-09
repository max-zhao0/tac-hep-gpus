#include <iostream>
#include <iomanip>

namespace mat {
    template <typename T>
    void fill(T* M, int width, T val) {
        for (int i = 0; i < width * width; i++) {
            M[i] = val;
        }
    }

    template <typename T>
    void pad(T* padded, const T* M, int width, int radius) {
        int padded_width = width + 2*radius;
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < width; j++) {
                padded[i + radius + padded_width * (j + radius)] = M[i + width * j];
                
                if (i < radius) {
                    padded[i + padded_width * (j + radius)] = 0;
                    padded[width + radius + i + padded_width * (j + radius)] = 0;
                }
                if (j < radius) {
                    padded[i + radius + padded_width * j] = 0;
                    padded[i + radius + padded_width* (width + radius + j)] = 0;
                }
                if (i < radius && j < radius) {
                    padded[i + padded_width *j] = 0;
                    padded[i + radius + width + padded_width * j] = 0;
                    padded[i + padded_width * (j + radius + width)] = 0;
                    padded[i + radius + width + padded_width * (j + radius + width)] = 0;
                }
            }
        }
    }

    template <typename T>
    void print(const T* mat, int width, int to_print_x=6, int to_print_y=6) {
        std::cout << "[";
        for (int i = 0; i < to_print_x; i++) {
            if (i > 0) {
                std::cout << " ";
            }

            std::cout << "[";
            for (int j = 0; j < to_print_y; j++) {
                std::cout << std::setw(5) << mat[i + width * j] << ", ";
            }
            std::cout << "... ]," << std::endl;
        }
        std::cout << " ... ]" << std::endl;
    }
}

namespace val {
    template <typename T>
    bool stencil_2d(T* stenciled, int width, int radius) {
        if (width < 2 * radius + 1) {
            std::cout << "Matrix too small to be checked with val::stencil_2d" << std::endl;
            return false;
        }

        if (
            (stenciled[0] != 2 * radius + 1) ||
            (stenciled[1] != 2 * radius + 2) ||
            (stenciled[1 + width * 1] != 2 * radius + 3) ||
            (stenciled[radius + width * radius] != 4 * radius + 1)
        ) return false;

        return true;
    }

    template <typename T>
    bool stenciled_squared(T* squared, int width, int radius) {
        if (width < 2 * radius + 1) {
            std::cout << "Matrix too small to be checked with val::stencil_2d" << std::endl;
            return false;
        }

        T max_top_row = 3 * radius + 1;
        T val00 = (width - 2 * radius) * max_top_row * max_top_row;
        for (int elem = max_top_row - radius; elem < max_top_row; elem++) {
            val00 += 2 * elem * elem;
        }

        T max_middle = 4 * radius + 1;
        T valrr = (width - 2 * radius) * max_middle * max_middle;
        for (int elem = max_middle - radius; elem < max_middle; elem++) {
            valrr += 2 * elem * elem;
        }

        if (
            (squared[0] != val00) ||
            (squared[radius + width * radius] != valrr)
        ) {
            std::cout << val00 << " != " << squared[0] << std::endl;
            std::cout << valrr << " != " << squared[radius + width * radius] << std::endl;
            return false;
        }

        return true;
    }
}
