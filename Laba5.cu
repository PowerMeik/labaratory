#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>
#include <ctime>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define ITER_TO_UPDATE 250

#define MAXIMUM_THREADS_PER_BLOCK 32

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Функция изменения матрицы
__global__ void iterate(double *A, double *A_new, size_t size){

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((j == 0) || (i == 0) || (i == size - 1) || (j == size - 1)) 
		return; // Don't update borders

    A_new[i * size + j] = 0.25 * (A[i * size + j - 1] + A[(i - 1) * size + j] + A[(i + 1) * size + j] + A[i * size + j + 1]);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Функция востановления границ матрицы
__global__ void restore(double* A, int size){
	size_t i = threadIdx.x;
	A[i] = 10.0 + i * 10.0 / (size - 1);
	A[i * size] = 10.0 + i * 10.0 / (size - 1);
	A[size - 1 + i * size] = 20.0 + i * 10.0 / (size - 1);
	A[size * (size - 1) + i] = 20.0 + i * 10.0 / (size - 1);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Функция разницы матриц
__global__ void subtraction(double* A, double* A_new, size_t size) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
	A_new[i * size + j] = A[i * size + j] - A_new[i * size + j];
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Значения по умодчанию
double eps = 1E-6;
int size = 512;
int iter_max = 1E6;

int main(int argc, char** argv) {
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Получение значений из командной строки
    for(int arg = 0; arg < argc; arg++){ 
        std::stringstream stream;
        if(strcmp(argv[arg], "-error") == 0){
            stream << argv[arg+1];
            stream >> eps;
        }
        else if(strcmp(argv[arg], "-iter") == 0){
            stream << argv[arg+1];
            stream >> iter_max;
        }
        else if(strcmp(argv[arg], "-size") == 0){
            stream << argv[arg+1];
            stream >> size;
        }
    }

	size_t totalSize = size * size;

	std::cout << "Settings: " << "\n\tMin error: " << eps << "\n\tMax iteration: " << iter_max << "\n\tSize: " << size << "x" << size << std::endl;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Выделения памяти
	cudaSetDevice(1);

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaGraph_t graph;
	cudaGraphExec_t graph_instance;

	dim3 blocksPerGrid = dim3(size / MAXIMUM_THREADS_PER_BLOCK, size / MAXIMUM_THREADS_PER_BLOCK);
    dim3 threadPerBlock = dim3(MAXIMUM_THREADS_PER_BLOCK, MAXIMUM_THREADS_PER_BLOCK);

	std::cout << "threadPerBlock " << MAXIMUM_THREADS_PER_BLOCK << "\nblocksPerGrid " << size / MAXIMUM_THREADS_PER_BLOCK << "\n";

	double *A_Device, *A_new_Device, *deviceError, *tempStorage = NULL;
	size_t tempStorageSize = 0;

	cudaMalloc(&A_Device, sizeof(double) * totalSize);
	cudaMalloc(&A_new_Device, sizeof(double) * totalSize);
	cudaMalloc(&deviceError, sizeof(double));

	restore<<<1, size>>>(A_Device, size);
	cudaMemcpy(A_new_Device, A_Device, sizeof(double) * totalSize, cudaMemcpyDeviceToDevice);

	cub::DeviceReduce::Max(tempStorage, tempStorageSize, A_new_Device, deviceError, totalSize, stream);
	cudaMalloc(&tempStorage, tempStorageSize);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Создание графа
	cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

	for (size_t i = 0; i < ITER_TO_UPDATE; i += 2) {
		iterate<<<blocksPerGrid, threadPerBlock, 0, stream>>>(A_Device, A_new_Device, size);
		iterate<<<blocksPerGrid, threadPerBlock, 0, stream>>>(A_new_Device, A_Device, size);
	}
	subtraction<<<blocksPerGrid, threadPerBlock, 0, stream>>>(A_Device, A_new_Device, size);
	cub::DeviceReduce::Max(tempStorage, tempStorageSize, A_new_Device, deviceError, totalSize, stream);
	restore<<<1, size, 0, stream>>>(A_new_Device, size);

	cudaStreamEndCapture(stream, &graph);
	cudaGraphInstantiate(&graph_instance, graph, NULL, NULL, 0);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Основной цикл
	int iter = 0; 
	double error = 1.0;
	clock_t begin = clock();
	while(iter < iter_max && error > eps) {
		cudaGraphLaunch(graph_instance, stream);
		cudaMemcpyAsync(&error, deviceError, sizeof(double), cudaMemcpyDeviceToHost);
		iter += ITER_TO_UPDATE;
	}
	clock_t end = clock();
	std::cout << "Result:\n\tIter: " << iter << "\n\tError: " << error << "\n\tTime: " << 1.0 * (end - begin) / CLOCKS_PER_SEC << std::endl;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// Чистка памяти
	cudaFree(A_Device);
	cudaFree(A_new_Device);
	cudaFree(tempStorage);
	cudaGraphDestroy(graph);
	cudaStreamDestroy(stream);
	return 0;
}