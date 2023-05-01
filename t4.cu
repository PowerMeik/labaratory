#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>



__global__ void error(double * A1, double *anew1 , double * res, int n){

     	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<n*n){
		res[idx] = fabs(anew1[idx]-A1[idx]);
	}
}

__global__ void compute(double * A1, double * anew1,int n){
	int j = blockIdx.x;						          
	int i = threadIdx.x;
	if((j>0) && (j<n-1) && (i>0) && (i<n-1))
		anew1[i + j * n] = 0.25 * (A1[i + j * n - n] + A1[i + j * n + n] + A1[i + j * n - 1] + A1[i + j * n + 1]);
}

int main(int argc, char** argv) {

	int iter_max;
	int sizeofnet1;
	double accuracy;

	accuracy = atof(argv[1]);
	sizeofnet1 = atoi(argv[2]);
	int n = sizeofnet1;
	iter_max = atoi(argv[3]);


	double* A = (double*)calloc(n * n, sizeof(double));
	double* anew = (double*)calloc(n * n, sizeof(double));



	A[0] = 10;
	A[0 + n - 1] = 20;
	A[0 + n * (n - 1)] = 20;
	A[(n - 1) * n + n - 1] = 30;
	anew[0] = 10;
	anew[0 + n - 1] = 20;
	anew[0 + n * (n - 1)] = 20;
	anew[(n - 1) * n + n - 1] = 30;

	double dif1 = (double)10 / (n - 1);
	for (int i = 1; i < n - 1; i++) {
		A[0 + i * n] = A[0 + i * n - n] + dif1;
		A[(n - 1) + i * n] = A[(n - 1) + i * n - n] + dif1;
		A[i] = A[i - 1] + dif1;
		A[i + (n - 1) * (n)] = A[i + (n - 1) * (n)-1] + dif1;
		anew[0 + i * n] = anew[0 + i * n - n] + dif1;
		anew[(n - 1) + i * n] = anew[(n - 1) + i * n - n] + dif1;
		anew[i] = anew[i - 1] + dif1;
		anew[i + (n - 1) * (n)] = anew[i + (n - 1) * (n)-1] + dif1;
	}
	int iter = 0;
       	void* t_memory = NULL;
	size_t t_memory_size = 0;
	double* A1 = NULL;
	double* anew1 = NULL;
	double* tmp_arr = NULL;
	double err = 100;
	double* err1 = 0;
	
	cudaMalloc((&err1), sizeof(double));
	cudaMalloc((&A1), sizeof(double) * (n * n));
	cudaMalloc((&anew1), sizeof(double) * (n*n));
	cudaMalloc((&tmp_arr), sizeof(double) *( n * n));
	cudaMemcpy(A1, A, sizeof(double) * (n * n), cudaMemcpyHostToDevice);
	cudaMemcpy(anew1, anew, sizeof(double) * (n * n), cudaMemcpyHostToDevice);
	
	cub::DeviceReduce::Max(t_memory, t_memory_size, tmp_arr, err1, n*n);
   	 cudaMalloc((&t_memory), t_memory_size);	
	for (iter = 0; iter < iter_max && err>accuracy; iter++) {
		compute<<<n-1, n-1>>>(A1, anew1, n);

		if (iter%100==0){
			error<<<n-1, n-1>>>(A1, anew1, tmp_arr, n);
			cub::DeviceReduce::Max(t_memory, t_memory_size, tmp_arr, err1, n*n);
			cudaMemcpy(&err, err1, sizeof(double), cudaMemcpyDeviceToHost);
		}
		double* tmp = A1;
		A1 = anew1;
		anew1 = tmp;
	}
	printf("%d\n%lf", iter, err);
	cudaFree(err1);
	cudaFree(t_memory);
	cudaFree(A1);
	cudaFree(anew1);
	free(A);
	free(anew);
	return 0;
}
