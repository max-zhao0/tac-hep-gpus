#include <iostream>

void swapValues(int& a, int& b) {
    int cache = b;
    b = a;
    a = cache;
}

void printArray(int* arr, int size) {
    std::cout << "[";
    for (int i = 0; i < size - 1; i++) {
        std::cout << arr[i] << ", ";
    }
    std::cout << arr[size - 1] << "]" << std::endl;
}

int main() {
    int size = 10;
    int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int B[] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

    printArray(A, size);
    printArray(B, size);

    for (int i = 0; i < size; i++) { 
        swapValues(A[i], B[i]);
    }

    std::cout << std::endl;
    printArray(A, size);
    printArray(B, size);
}