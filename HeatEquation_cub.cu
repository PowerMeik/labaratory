#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>


__global__
void getErrorMatrix(double* u, double* u_new, double* end){
    
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(!(blockIdx.x == 0 || threadIdx.x == 0)){
		end[idx] = std::abs(u_new[idx] - u[idx]);
	}
}
__global__
void solve(double* u, double* u_new, size_t size){
 
	size_t i = blockIdx.x;
	size_t j = threadIdx.x;	
	if(!(blockIdx.x == 0 || threadIdx.x == 0)){
		u_new[i * size + j] = 0.25 * (u[i * size + j - 1] + 
                                u[(i - 1) * size + j] +
                                u[(i + 1) * size + j] + 
                                u[i * size + j + 1]);		
	}
}

int main(int argc, char** argv) {
      //int  iters = 1000000, length = 128;
    //double accuracy = 1e-6;
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " epsilon N maxIter" << std::endl;
        return 1;
    }
    int iters = std::stoi(argv[1]);
        //std::cout<<iters<<std::endl;
    int  length = std::stoi(argv[2]);
        //std::cout<<grid_size<<std::endl;
    double accuracy = std::stod(argv[3]);
        //std::cout<<accuracy<<std::endl;

    int u_size = length * length;
    double step = 1.0 * (20 - 10) / (length - 1);
    auto* u = new double[u_size];
    auto* u_new = new double[u_size];
    std::memset(u, 0, sizeof(double) * u_size);
  
    u[0] = 10;
    u[length - 1] = 20;
    u[u_size - 1] = 30;
    u[length * (length - 1)] = 20;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < length - 1; i++) {
        u[i] = 10 + i * step;
        u[i * (length)] = 10 + i * step;
        u[length * i + (length - 1)] = 20 + i * step;
        u[length * (length - 1) + i] = 20 + i * step;
    }
    std::memcpy(u_new, u, sizeof(double) * u_size);
    double error = 1.0, min_error = accuracy;
    int max_iter = iters, iter = 0;

    double* d_u, *d_u_new, *deviceError, *errMx, *buff = NULL;
	size_t sizeofBuff = 0;    

    cudaError_t stat;
    stat = cudaMalloc((void**)(&d_u), sizeof(double) * u_size);
    stat = cudaMalloc((void**)(&d_u_new), sizeof(double) * u_size);
    stat = cudaMalloc((void**)&errMx, sizeof(double) * u_size);

    cudaMalloc((void**)&deviceError, sizeof(double));

    stat = cudaMemcpy(d_u, u, sizeof(double) * u_size, cudaMemcpyHostToDevice);
    stat = cudaMemcpy(d_u_new, u_new, sizeof(double) * u_size, cudaMemcpyHostToDevice);

	cub::DeviceReduce::Max(buff, sizeofBuff, errMx, deviceError, u_size);
	cudaMalloc((void**)&buff, sizeofBuff);
    
    for (iter = 0; iter < max_iter; iter++)
    {
        solve << <length - 1, length - 1 >> > (d_u, d_u_new, length);
        if (iter % 100 == 0) {

            getErrorMatrix << <length - 1, length - 1 >> > (d_u, d_u_new, errMx);

            cub::DeviceReduce::Max(buff, sizeofBuff, errMx, deviceError, u_size);

            stat = cudaMemcpy(&error, deviceError, sizeof(double), cudaMemcpyDeviceToHost);
            if (error < min_error)
                break;
            
        }
        std::swap(d_u, d_u_new);
    }
	/*while(iter < max_iter && error > min_error){
		iter++;
    
    solve<<<length - 1, length - 1>>>(d_u, d_u_new, length);
		if(iter % 100 == 0){
            
			getErrorMatrix<<<length - 1, length - 1>>>(d_u, d_u_new, errMx);
            
			cub::DeviceReduce::Max(buff, sizeofBuff, errMx, deviceError, u_size);
       
			cudaMemcpy(&error, deviceError, sizeof(double), cudaMemcpyDeviceToHost);
		}
		std::swap(d_u, d_u_new);
	}*/
 
    auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Iteration count: " << iter << std::endl;
    std::cout << "Accuracy: " << error << std::endl;
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << "\n";

    free(u);    
    free(u_new);
	cudaFree(d_u);
	cudaFree(d_u_new);
	cudaFree(errMx);
	cudaFree(buff);
    return 0;
}