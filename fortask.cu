#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void heat_equation(double* u, double* u_new, double dt, int ny) {
    size_t i = blockIdx.x;
	size_t j = threadIdx.x;
    
	if(!(blockIdx.x == 0 || threadIdx.x == 0))
	{
        u_new[i * ny + j] = dt * (u[(i - 1) * ny + j] + u[(i + 1) * ny + j]+
            u[i * ny + (j - 1)] + u[i * ny + (j + 1)]);
    }
}

__global__ void get_error(double* u, double* u_new, double* out)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx>0)
	{
		out[idx] = fabs(u_new[idx] - u[idx]);
	}
}


int main(int argc, char** argv) {
    double accuracy;
    int size, iternum;
    accuracy = atof(argv[1]);
    size = atoi(argv[2]);
    iternum = atoi(argv[3]);

   double* array = (double*)calloc(size * size, sizeof(double));
   double* arraynew = (double*)calloc(size * size, sizeof(double));

    array[0] = 10.0;
    array[size-1] = 20.0;
    array[(size-1) *size] = 20.0;
    array[size * size-1] = 30.0;

    double error = 1.0;
    double step = 10.0/(size-1);
    size_t realsize = size*size;


    clock_t start = clock();

    for (int i = 1; i < size-1; i++) {
        array[i] = array[0] + step * i;
        array[size * (size - 1) + i] = array[(size - 1) * size] + step*i;
        array[size * i] = array[0] + step*i;
        array[(size - 1) + i * size] = array[size - 1] + step * i;
    }

    memcpy(arraynew, array, sizeof(double) * realsize);

    double* Matrix, *MatrixNew, *Error, *deviceError, *errortemp = 0;

    cudaMalloc((&Matrix), realsize*sizeof(double));
    cudaMalloc((&MatrixNew), realsize*sizeof(double));
    cudaMalloc((&Error), realsize*sizeof(double));
    cudaMalloc((&deviceError), sizeof(double));
    
    cudaMemcpy(Matrix, array, realsize*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(MatrixNew, arraynew, realsize*sizeof(double), cudaMemcpyHostToDevice);

    size_t tempsize = 0;
    double dt = 0.25;
    int k = 0;

    cub::DeviceReduce::Max(errortemp, tempsize, Error, deviceError, realsize);
    cudaMalloc((&errortemp), tempsize);

    for (; (k < iternum) && (error > accuracy); ++k) { 
        heat_equation<<<size-1, size-1>>>(Matrix, MatrixNew, dt, size);
        if(k%100==0){
            get_error<<<size-1, size-1>>>(Matrix, MatrixNew, Error);
            cub::DeviceReduce::Max(errortemp, tempsize, Error, deviceError, realsize);
            cudaMemcpy(&error, deviceError, sizeof(double), cudaMemcpyDeviceToHost);
            }
        double* temp = Matrix;
        Matrix = MatrixNew;
        MatrixNew = temp;
    }

    clock_t end = clock();
    printf("Time is = %lf\n", 1.0*(end-start)/CLOCKS_PER_SEC);
    
    printf("%d %lf\n", k, error);

    cudaFree(Matrix);
    cudaFree(MatrixNew);
    free(array);
    free(arraynew);
    return 0;

}