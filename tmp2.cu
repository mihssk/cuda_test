#include <iostream>
#include "cuda_runtime.h"
int main()
{
    int b = 4;
    int c = 5;
    int* a[2] = {&b, &c};

    std::cout << *a[1];


    return 0;
}