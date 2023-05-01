#include <iostream>
//#include <cstring>
#include <sstream>


#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>

#include "cuda_runtime.h"

//#include "sub.cuh"

#define IDX2C(i, j, ld) (((j)*(ld))+(i))


__global__ void heat_equation(double* arr, double* arr2, int N) {
    size_t i = blockIdx.x;
	size_t j = threadIdx.x;
    
	if(!(blockIdx.x == 0 || threadIdx.x == 0))
	{
        arr2[IDX2C(i, j, N)] = (arr[IDX2C(i + 1, j, N)] + arr[IDX2C(i - 1, j, N)] + arr[IDX2C(i, j - 1, N)] + arr[IDX2C(i, j + 1, N)]) * 0.25;
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


void print_array(double *A, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            printf("%.6f\t", A[IDX2C(i, j, size)]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


template<typename T>
T extractNumber(char* arr){
    std::stringstream stream;
    stream << arr;
    T result;
    if (!(stream >> result)){
        throw std::invalid_argument("Wrong argument type");
    }
    return result;
}


int main(int argc, char *argv[]) {
    cudaSetDevice(2);

    int N = 128;
    int num_of_iter = 1000000;
    double accuracy = 0.000001;

//Получаем параметры из командной строки
    for(int arg = 0; arg < argc; arg++){
        if(std::strcmp(argv[arg], "-eps") == 0){
            accuracy = extractNumber<double>(argv[arg+1]);
            arg++;
        }
        else if(std::strcmp(argv[arg], "-i") == 0){
            num_of_iter = extractNumber<int>(argv[arg+1]);
            arg++;
        }
        else if(std::strcmp(argv[arg], "-s") == 0){
            N = extractNumber<int>(argv[arg+1]);
            arg++;
        }
    }

//Начинаем отсчет времени
    clock_t start = clock();
    
//Объявляем массивы
    int size = N*N*sizeof(double);
    double *arr = (double *)calloc(sizeof(double), size);
    double *arr2 = (double *)calloc(sizeof(double), size);
    double *arr3 = (double *)calloc(sizeof(double), size);

    double delta = 10.0 / (N-1);

//Заполняем массив
    arr[IDX2C(0, 0, N)] = 10.0;
    arr[IDX2C(0, N - 1, N)] = 20.0;
    arr[IDX2C(N - 1, 0, N)] = 20.0;
    arr[IDX2C(N - 1, N - 1, N)] = 30.0;

    for(int i = 1; i < N - 1; i++){
        arr[IDX2C(0, i, N)] = arr[IDX2C(0, i-1, N)] + delta;
        arr[IDX2C(N - 1, i, N)] = arr[IDX2C(N - 1, i-1, N)] + delta;
        arr[IDX2C(i, 0, N)] = arr[IDX2C(i-1, 0, N)] + delta;
        arr[IDX2C(i, N - 1, N)] = arr[IDX2C(i-1, N - 1, N)] + delta;
    }
    memcpy(arr2, arr, size);

//Объявляем и заполняем массивы на cuda
    double* Matrix, *MatrixNew, *Error, *deviceError, *errortemp = 0;

    cudaMalloc((&Matrix), size);
    cudaMalloc((&MatrixNew), size );
    cudaMalloc((&Error), size );  
    cudaMalloc((&deviceError), sizeof(double));
    
    cudaMemcpy(Matrix, arr, size , cudaMemcpyHostToDevice);
    cudaMemcpy(MatrixNew, arr2, size , cudaMemcpyHostToDevice);
   
    size_t tempsize = 0;
    int k = 0;
    double error = 30;

    cub::DeviceReduce::Max(errortemp, tempsize, Error, deviceError, size);
    cudaMalloc((&errortemp), tempsize);

    for (; (k < num_of_iter) && (error > accuracy); k++) { 
        //Перезаполняем массив
        heat_equation<<<N-1, N-1>>>(Matrix, MatrixNew, N);
        if(k % 10 == 0){
            //Вычисляем матрицу ошибок
            get_error<<<N, N>>>(Matrix, MatrixNew, Error);
            //Вычисляем максимальную ошибку
            cub::DeviceReduce::Max(errortemp, tempsize, Error, deviceError, N*N);
            //Копируем ошибку на устройство
            cudaMemcpy(&error, deviceError, sizeof(double), cudaMemcpyDeviceToHost);
            //printf("%lf\n", error);
            }
        //Обновляем массив
        double* temp = Matrix;
        Matrix = MatrixNew;
        MatrixNew = temp;
    }

//Заканчиваем считать время
    clock_t end = clock();
    printf("%lf\n", 1.0*(end-start)/CLOCKS_PER_SEC);
    
    printf("%d\n%lf\n", k, error);
    std::cout<<"Error"<<error;

//Очищаем память
    cudaFree(Matrix);
    cudaFree(MatrixNew);
    free(arr);
    free(arr2);
    return 0;
}