// подключение библиотек С++
#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define at(arr, x, y) (arr[(x) * size + (y)]) 

double eps = 1E-6;
int iter_max = 1E6;
int size = 128;
bool mat = false;

__global__ void calc(double* A, double* B)
{
    int i = blockIdx.x + 1;
    int j = threadIdx.x + 1;
    at(B, i, j) = 0.25 * ( at(A, i, j+1) + at(A, i, j-1) + at(A, i-1, j) + at(A, i+1, j) );
}

__global__ void sub(double* A, double* B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    B[i] = A[i] - B[i];
}

int main(int argc, char **argv)
{
    // ввод данных из консоли
    for(int arg = 0; arg < argc; arg += 1)
    { 
        if(strcmp(argv[arg], "-error") == 0)
        {   
            // ошибка
            eps = atof(argv[arg+1]);
            arg += 1;
        }
        else if(strcmp(argv[arg], "-iter") == 0)
        {
            // итерации
            iter_max = atoi(argv[arg+1]);
            arg += 1;
        }
        else if(strcmp(argv[arg], "-size") == 0)
        {
            // размер
            size = atoi(argv[arg+1]);
            arg += 1;
        }
        else if(strcmp(argv[arg], "-mat") == 0)
        {
            mat = true;
            arg += 1;
        }
    }

    double* A, Anew;

    cudaMalloc(&A, size);
    cudaMalloc(&Anew, size);

    for(int i = 0; i < size; i += 1) 
    {
        at(A, 0, i)        = 10.0 / (size - 1) * i + 10;
        at(A, i, 0)        = 10.0 / (size - 1) * i + 10;
        at(A, size - 1, i) = 10.0 / (size - 1) * i + 20;
        at(A, i, size - 1) = 10.0 / (size - 1) * i + 20;
    }

    double error  = 1.0;
    int iteration = 0;



    return 0;
}