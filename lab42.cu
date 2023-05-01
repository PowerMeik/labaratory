#include <iostream>
#include <string>
#include <cmath>
#include <ctime> 
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda.h>

using namespace std;

__global__
void countnewmatrix(double* mas, double* anew, size_t n)
{
    size_t i = blockIdx.x;
    size_t j = threadIdx.x;

    if (!(blockIdx.x == 0 || threadIdx.x == 0))
        anew[i * n + j] = (mas[i * n + j - 1] + mas[(i - 1) * n + j] + mas[(i + 1) * n + j] + mas[i * n + j + 1]) * 0.25;

}


__global__
void finderr(double* mas, double* anew, double* outMatrix)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!(blockIdx.x == 0 || threadIdx.x == 0))
        outMatrix[idx] = fabs(anew[idx] - mas[idx]);

}

int main(int arg, char** argv) {
    int N = atoi(argv[1]);
    int ITER = atoi(argv[2]);
    float ACC = atof(argv[3]);
    
    if (N == 0 || N < 0){
	std::cout<<"N error"<<std::endl;
	return EXIT_FAILURE;
    }

    if (ITER > 1000000 || ITER < 0)
    {
	std::cout<<"ITER error"<<std::endl;
	return EXIT_FAILURE;
    }

    if (ACC < 0.0000001){
	std::cout<<"ACC error"<<std::endl;
	return EXIT_FAILURE;
    }


    double* mas = new double[N * N];
    double* anew = new double[N * N];

    for (int i = 0; i < N * N; i++)
    {
        mas[i] = 0;
        anew[i] = 0;
    }

    int rep = 0;
    double err = 1.0;

    mas[0] = 10;
    mas[N - 1] = 30;
    mas[N * N - 1] = 20;
    mas[N * (N - 1)] = 20;

    anew[0] = 10;
    anew[N - 1] = 30;
    anew[N * (N - 1)] = 20;
    anew[N * N - 1] = 20;

    clock_t befin = clock();

    for (int i = 1; i < N-1; i++)
        mas[i] = mas[i-1] + (mas[N - 1] - mas[0]) / (N-1);

    for (int i = 1; i < N-1; i++)
    {
        mas[N * (N - 1) + i] = mas[N * (N - 1) + i-1] + (mas[N * N-1] - mas[N * (N - 1)]) / (N-1);
        mas[N*i] = mas[N*i-N] + (mas[N * N-1] - mas[N-1]) / (N-1);
        mas[(N)*i+(N-1)] = mas[(N)*(i-1)+(N-1)] + (mas[N * N - 1] - mas[N - 1]) / (N-1);
    }
  
	for (int i = 0; i < N*N; i++)
		anew[i] = mas[i];

        cout << "Initialization Time: " << 1.0 * (clock() - befin) / CLOCKS_PER_SEC << endl;
        clock_t befca = clock();

        cudaSetDevice(3);

        double* mas_dev, * anew_dev, * deviceError, * errorMatrix, * tempStorage = NULL;
        size_t tempStorageSize = 0;

        cudaMalloc((void**)(&mas_dev), sizeof(double) * N * N);
        cudaMalloc((void**)(&anew_dev), sizeof(double) * N * N);
        cudaMalloc((void**)&deviceError, sizeof(double));
        cudaMalloc((void**)&errorMatrix, sizeof(double) * N * N);

        cudaMemcpy(mas_dev, mas, sizeof(double) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(anew_dev, anew, sizeof(double) * N * N, cudaMemcpyHostToDevice);

        cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, N * N);
        cudaMalloc(&tempStorage, tempStorageSize);


        while ((rep < ITER) && (err >= ACC))
        {
            rep++;
            countnewmatrix <<<N - 1, N - 1>>> (mas_dev, anew_dev, N);

            if (rep % 100 == 0)
            {
                finderr <<<N - 1, N - 1>>> (mas_dev, anew_dev, errorMatrix);
                cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, N * N);
                cudaMemcpy(&err, deviceError, sizeof(double), cudaMemcpyDeviceToHost);
            }


            double* c = mas_dev;
            mas_dev = anew_dev;
            anew_dev = c;
            cout << rep << "  " << err << endl;
        }

    cout << "Calculation time: " << 1.0 * (clock() - befca) / CLOCKS_PER_SEC << endl;
    cout << "Iteration: " << rep << " " << "Error: " << err << endl;
    delete[] mas;
    delete[] anew;
    cudaFree(mas_dev);
    cudaFree(anew_dev);
    cudaFree(errorMatrix);
    cudaFree(tempStorage);
    return EXIT_SUCCESS;
}
