#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cub/cub.cuh>

// calculate "a_new" matrix
__global__ void calculate_matrix(double* a, double* a_new, int size)
{
    int i = blockIdx.x;
    int j = threadIdx.x;

    if (i != 0 && j != 0)
    {
        a_new[i * size + j] = 0.25 * (a[i * size + j - 1] +
                                      a[i * size + j + 1] +
                                      a[(i - 1) * size + j] +
                                      a[(i + 1) * size + j]);
    }
}

// calculate substraction of matrices (errors, matrix of errors)
__global__ void calculate_error_matrix(double* a, double* a_new, double* err_matrix)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x != 0 && threadIdx.x != 0)
    {
        err_matrix[i] = fabs(a_new[i] - a[i]);
    }
}

// <program name> size tol iter_max by launching
int main(int argc, char** argv)
{
    // variables from cmd
    int size = atoi(argv[1]);
    double tol = atof(argv[2]);
    double iter_max = atof(argv[3]);

    int iter = 0;
    double step = 10.0 / size;
    double err = 1.0;

    // "corner" indices
    int up_left = 0;
    int up_right = size - 1;
    int down_left = size * size - size;
    int down_right = size * size - 1;

    // our "matrices" and temp variable for swap
    double* temp;
    double* a = (double*)calloc(size * size, sizeof(double));
    double* a_new = (double*)calloc(size * size, sizeof(double));

    // fill "corner" values
    a[up_left] = 10;
    a[up_right] = 20;
    a[down_right] = 30;
    a[down_left] = 20;
    a_new[up_left] = 10;
    a_new[up_right] = 20;
    a_new[down_right] = 30;
    a_new[down_left] = 20;   

    // transpose up_left and up_right
    for (int i = 1; i < up_right; ++i)
    {
        a[i] = a[i - 1] + step;
	    a_new[i] = a[i];
    }
    // transpose down_left and down_right
    for (int i = down_left + 1; i < down_right; ++i)
    {
        a[i] = a[i - 1] + step;
	    a_new[i] = a[i];
    }
    // transpose up_left and down_left, up_right and down_right
    for (int i = size; i < down_left; i += size)
    {
        a[i] = a[i - size] + step;
        a[i + size - 1] = a[i - 1] + step;
	    a_new[i] = a[i];
	    a_new[i + size - 1] = a[i + size - 1];
    }


    // cuda part

    // choose device
    cudaSetDevice(1);

    // device copy variables
    double* dev_a = NULL;
    double* dev_a_new = NULL;
    double* dev_err = NULL;
    // substraction of matrices
    double* err_matrix = NULL;
    // temporary storage for reduce (and its size)
    double* temp_storage = NULL;
    size_t temp_storage_size = 0;

    // allocate memory on device
    cudaMalloc((void**)&dev_a, sizeof(double) * size * size);
    cudaMalloc((void**)&dev_a_new, sizeof(double) * size * size);
    cudaMalloc((void**)&err_matrix, sizeof(double) * size * size);
    cudaMalloc((void**)&dev_err, sizeof(double));

    // determine "temp_storage_size" with cub function and allocate memory
    cub::DeviceReduce::Max(temp_storage, temp_storage_size, err_matrix, dev_err, size * size);
    cudaMalloc((void**)&temp_storage, temp_storage_size);

    // copy data into device
    cudaMemcpy(dev_a, a, sizeof(double) * size * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_a_new, a_new, sizeof(double) * size * size, cudaMemcpyHostToDevice);


    clock_t begin = clock();

    // main cycle
    while (err > tol && iter < iter_max)
    {
        iter++;

        // calculate "a_new" matrix
        calculate_matrix<<<size - 1, size - 1>>>(dev_a, dev_a_new, size);

        if (iter % 100 == 0)
        {
            // calculate matrix of errors, then find max error and copy its value to "err" stored on CPU
            calculate_error_matrix<<<size - 1, size - 1>>>(dev_a, dev_a_new, err_matrix);
            cub::DeviceReduce::Max(temp_storage, temp_storage_size, err_matrix, dev_err, size * size);
            cudaMemcpy(&err, dev_err, sizeof(double), cudaMemcpyDeviceToHost);
        }

        temp = dev_a;
        dev_a = dev_a_new;
        dev_a_new = temp;
    }

    clock_t end = clock();

    printf("%d:\t%-32.25lf\n", iter, err);
    printf("Time:\t %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);

    cudaFree(dev_a);
    cudaFree(dev_a_new);
    cudaFree(err_matrix);
    cudaFree(dev_err);
    cudaFree(temp_storage);

    free(a);
    free(a_new);

    return 0;
}
