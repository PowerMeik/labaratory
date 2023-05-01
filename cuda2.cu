#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <nvtx3/nvToolsExt.h>

__global__
void getErrorMatrix(double* A, double* Anew, double* end){
 
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(!(blockIdx.x == 0 || threadIdx.x == 0)){
		end[idx] = std::abs(Anew[idx] - A[idx]);
	}
}
__global__
void calculateMatrix(double* A, double* Anew, size_t size){
    
	size_t i = blockIdx.x;
	size_t j = threadIdx.x;	
	if(!(blockIdx.x == 0 || threadIdx.x == 0)){
		Anew[i * size + j] = 0.25 * (A[i * size + j - 1] + 
                                A[(i - 1) * size + j] +
                                A[(i + 1) * size + j] + 
                                A[i * size + j + 1]);		
	}
}



int main(int argc, char** argv) {
   // int  iters = 1000000, grid_size = 128;
    //double accuracy = 1e-6;
       if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " epsilon N maxIter" << std::endl;
        return 1;
    }
    int iters = std::stoi(argv[1]);
        //std::cout<<iters<<std::endl;
    int  grid_size = std::stoi(argv[2]);
        //std::cout<<grid_size<<std::endl;
    double accuracy = std::stod(argv[3]);
        //std::cout<<accuracy<<std::endl;
    
    int full_size = grid_size * grid_size;
    double step = 1.0 * (20 - 10) / (grid_size - 1);
    auto* A = new double[full_size];
    auto* Anew = new double[full_size];
    std::memset(A, 0, sizeof(double) * full_size);
   
    A[0] = 10;
    A[grid_size - 1] = 20;
    A[full_size - 1] = 30;
    A[grid_size * (grid_size - 1)] = 20;

   
    for (int i = 1; i < grid_size - 1; i++) {
        A[i] = 10 + i * step;
        A[i * (grid_size)] = 10 + i * step;
        A[grid_size * i + (grid_size - 1)] = 20 + i * step;
        A[grid_size * (grid_size - 1) + i] = 20 + i * step;
    }
    std::memcpy(Anew, A, sizeof(double) * full_size);
    double error = 1.0, min_error = accuracy;
    int max_iter = iters, iter = 0;


    double* ptr_A, *ptr_Anew, *deviceError, *errMx, *buff = NULL;
	size_t sizeofBuff = 0;    
       cudaError_t cudaStatus_1 = cudaMalloc((void**)(&ptr_A), sizeof(double) * full_size);
	cudaError_t cudaStatus = cudaMalloc((void**)(&ptr_Anew), sizeof(double) * full_size);
	cudaStatus_1 = cudaMalloc((void**)&errMx, sizeof(double) * full_size);
	if (cudaStatus_1 != 0 || cudaStatus != 0){
		std::cout << "Pu-pu-pu, something is wrong with memory allocation" << std::endl;
		return 42;
	}    
       cudaMalloc((void**)&deviceError, sizeof(double));

	cudaStatus_1 = cudaMemcpy(ptr_A, A, sizeof(double) * full_size, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(ptr_Anew, Anew, sizeof(double) * full_size, cudaMemcpyHostToDevice);
	if (cudaStatus_1 != 0 || cudaStatus != 0){
		std::cout << "Pu-pu-pu, something is wrong with memory transfer" << std::endl;
		return 42;
	}

	
	cub::DeviceReduce::Max(buff, sizeofBuff, errMx, deviceError, full_size);
	cudaMalloc((void**)&buff, sizeofBuff);

    nvtxRangePushA("pepe");
	while(iter < max_iter && error > min_error){
		iter++;
    calculateMatrix<<<grid_size - 1, grid_size - 1>>>(ptr_A, ptr_Anew, grid_size);
		if(iter % 100 == 0){
           			getErrorMatrix<<<grid_size - 1, grid_size - 1>>>(ptr_A, ptr_Anew, errMx);
            
            			cub::DeviceReduce::Max(buff, sizeofBuff, errMx, deviceError, full_size);
            
			cudaMemcpy(&error, deviceError, sizeof(double), cudaMemcpyDeviceToHost);
		}
		std::swap(ptr_A, ptr_Anew);
	}
   
	std::cout << "Iter: " << iter << std::endl;
    std::cout << "Error: " << error << std::endl;

 
    free(A);    
    free(Anew);
	cudaFree(ptr_A);
	cudaFree(ptr_Anew);
	cudaFree(errMx);
	cudaFree(buff);
    return 0;
}