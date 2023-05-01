#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cub/cub.cuh>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void initMatrix(
    double* A, double* Anew,
    int netSize, double hst, double hsb, double vsl, double vsr,
    double tl, double tr, double bl, double br)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    A[i * netSize] = vsl * i + tl;
    A[i] = hst * i + tl;
    A[((netSize - 1) - i) * netSize + (netSize - 1)] = vsr * i + br;
    A[(netSize - 1) * netSize + ((netSize - 1) - i)] = hsb * i + br;

    Anew[i * netSize] = vsl * i + tl;
    Anew[i] = hst * i + tl;
    Anew[((netSize - 1) - i) * netSize + (netSize - 1)] = vsr * i + br;
    Anew[(netSize - 1) * netSize + ((netSize - 1) - i)] = hsb * i + br;
}

void printMatrix(double* A, int netSize)
{
    std::cout << "netSize: " << netSize << std::endl;
    for (int i = 0; i < netSize; i++)
    {
        for (int j = 0; j < netSize; j++)
        {
            std::cout << A[i * netSize + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

__global__ void iterateMatrix(double* A, double* Anew, int netSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || i > netSize - 1 || j < 1 || j > netSize - 1) return;
    int index = i * netSize + j;

    Anew[index] = 0;
    int div_amount = 0;
    if (i - 1 >= 0) {
        Anew[index] += A[index - netSize];
        div_amount++;
    }

    if (i + 1 < netSize) {
        Anew[index] += A[index + netSize];
        div_amount++;
    }

    if (j - 1 >= 0) {
        Anew[index] += A[index - 1];
        div_amount++;
    }

    if (j + 1 < netSize) {
        Anew[index] += A[index + 1];
        div_amount++;
    }

    if (div_amount == 0) return;
    Anew[i * netSize + j] /= div_amount;
}

__global__ void findDifference(double* A, double* Anew) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    A[i] = Anew[i] - A[i];
}

__global__ void swapMatrix(double* A, double* Anew)
{
    double* temp = A;
    A = Anew;
    Anew = temp;
}

int main(int argc, char* argv[])
{
    dim3 threadsPerBlock = (16, 16);
    printf("threads per block - (%d %d)\n", threadsPerBlock.x, threadsPerBlock.y);

    int netSize = 128;
    double minError = 0.000001;
    int maxIterations = 1000000;
    char* end;
    if (argc != 4) {
        std::cout << "You must enter exactly 3 arguments:\n1. Grid size (one number)\n2. Minimal error\n3. Iterations amount\n";
        return -1;
    }
    else {
        netSize = strtol(argv[1], &end, 10);
        minError = strtod(argv[2], &end);
        maxIterations = strtol(argv[3], &end, 10);
    }
    std::cout << netSize << " " << minError << " " << maxIterations << std::endl;

    //values of net edges
    const double tl = 10, //top left
        tr = 20, //top right
        bl = 20, //bottom left
        br = 30; //bottom right

    const double hst = (tr - tl) / (netSize - 1), //horizontal step top
        hsb = (bl - br) / (netSize - 1), //horizontal step bottom
        vsl = (bl - tl) / (netSize - 1), //vertical step left
        vsr = (tr - br) / (netSize - 1); //vertical step right

    double* A_h = (double*)malloc(sizeof(double*) * netSize * netSize);
    double* A_d;
    double* Anew;
    double* max;

    printf("threads per block - (%d %d)\n", threadsPerBlock.x, threadsPerBlock.y);

    gpuErrchk( cudaMalloc(&A_d, sizeof(double*) * netSize * netSize) );
    gpuErrchk( cudaMalloc(&Anew, sizeof(double*) * netSize * netSize) );
    gpuErrchk( cudaMalloc(&max, sizeof(double)) );

    gpuErrchk( cudaMemset(A_d, 0, sizeof(double) * netSize) );

    void* d_tempStorage = NULL;
    size_t d_tempStorageBytes = 0;
    cub::DeviceReduce::Max(d_tempStorage, d_tempStorageBytes, A_d, max, netSize*netSize);

    initMatrix <<< netSize / threadsPerBlock.x, threadsPerBlock.x >>> (A_d, Anew, netSize, hst, hsb, vsl, vsr, tl, tr, bl, br);
    gpuErrchk( cudaGetLastError() );
    std::cout << "init:\n";
    gpuErrchk( cudaMemcpy(A_h, A_d, sizeof(double) * netSize * netSize, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaDeviceSynchronize() )
    printMatrix(A_h, netSize);

    double error = 10.;
    int iteration = 0;
    dim3 blockNum = ((int)(netSize / threadsPerBlock.x), (int)(netSize / threadsPerBlock.y));
    printf("block amount - (%d %d)\n", blockNum.x, blockNum.y);
    printf("threads per block - (%d %d)\n", threadsPerBlock.x, threadsPerBlock.y);

    while (error > minError && iteration < maxIterations)
    {
        iterateMatrix <<< blockNum, threadsPerBlock >>> ( A_d, Anew, netSize );
        swapMatrix <<< 1, 1 >>> (A_d, Anew);
        gpuErrchk( cudaGetLastError() );

        //every 100 iteration will be documented
        if (iteration % 100 == 0) {
            
            gpuErrchk( cudaMemcpy(A_h, Anew, sizeof(double) * netSize * netSize, cudaMemcpyDeviceToHost) );
            gpuErrchk(cudaDeviceSynchronize())
            printMatrix(A_h, netSize);

            findDifference <<< (int)((netSize * netSize) / threadsPerBlock.x), threadsPerBlock.x >>> (A_d, Anew);
            cub::DeviceReduce::Max(d_tempStorage, d_tempStorageBytes, A_d, max, netSize * netSize);
            gpuErrchk( cudaMemcpy(&error, max, sizeof(double), cudaMemcpyDeviceToHost) );

            std::cout << "iteration " << iteration + 1 << "/" << maxIterations << " error = " << error << "\t(min " << minError << ")" << std::endl;
            gpuErrchk( cudaGetLastError() );
        }
        ++iteration;
    }
    //after completion copy matrix to CPU
    gpuErrchk(cudaMemcpy(A_h, Anew, sizeof(double) * netSize * netSize, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize())
    printMatrix(A_h, netSize); //print matrix (debug feature)

    findDifference <<< (int)((netSize * netSize) / threadsPerBlock.x), threadsPerBlock.x >>> (A_d, Anew);
    cub::DeviceReduce::Max(d_tempStorage, d_tempStorageBytes, A_d, max, netSize * netSize);
    gpuErrchk(cudaMemcpy(&error, max, sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "Program ended. " << " Final error = " << error << "\t(min " << minError << ")" << std::endl;
    gpuErrchk(cudaGetLastError());

    cudaFree(&A_d);
    cudaFree(&Anew);

    return 0;
}
