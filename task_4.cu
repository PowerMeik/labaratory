#include <chrono>
#include <cmath>
#include <iostream>
#include "cuda_runtime.h"
#include <cub/cub.cuh>

#ifdef _FLOAT
    #define T float
    #define MAX std::fmaxf
    #define STOD std::stof
#else
    #define T double
    #define MAX std::fmax
    #define STOD std::stod
#endif

// Макрос индексации с 0
#define IDX2C(i, j, ld) (((j)*(ld))+(i))

// Вывести значения двумерного массива
void print_array(T *A, int size)
{
    for (int i = 0; i < size * size - 1; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            // Значение с GPU
            //#pragma acc kernels present(A)
            printf("%.2f\t", A[IDX2C(i, j, size)]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


__global__ void calculateMatrix(double* A, double* Anew, uint32_t size)
{
    if(blockIdx.x == 0 || threadIdx.x == 0) return;

	uint32_t i = blockIdx.x;
	uint32_t j = threadIdx.x;
	Anew[IDX2C(i, j, size)] = (A[IDX2C(i + 1, j, size)] + A[IDX2C(i - 1, j, size)]
                                        + A[IDX2C(i, j - 1, size)] + A[IDX2C(i, j + 1, size)]) * 0.25;	
}

__global__ void getErrorMatrix(double* matrixA, double* matrixB, double* outputMatrix)
{
	if(blockIdx.x == 0 || threadIdx.x == 0) return;

	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	outputMatrix[idx] = std::abs(matrixB[idx] - matrixA[idx]);
}


// Инициализация матрицы, чтобы подготовить ее к основному алгоритму
void initialize_array(T *A, int size)
{
    // Заполнение углов матрицы значениями
    A[IDX2C(0, 0, size)] = 10.0;
    A[IDX2C(0, size - 1, size)] = 20.0;
    A[IDX2C(size - 1, 0, size)] = 20.0;
    A[IDX2C(size - 1, size - 1, size)] = 30.0;

    // Заполнение периметра матрицы
    T step = 10.0 / (size - 1);

    for (int i = 1; i < size - 1; ++i)
    {
        T addend = step * i;
        A[IDX2C(0, i, size)] = A[IDX2C(0, 0, size)] + addend;                 // horizontal
        A[IDX2C(size - 1, i, size)] = A[IDX2C(size - 1, 0, size)] + addend;   // horizontal
        A[IDX2C(i, 0, size)] = A[IDX2C(0, 0, size)] + addend;                 // vertical
        A[IDX2C(i, size - 1, size)] = A[IDX2C(0, size - 1, size)] + addend;   // vertical
    }
}

// Основной алгоритм
void calculate(int net_size = 128, int iter_max = 1e6, T accuracy = 1e-6, bool res = false)
{
    cudaSetDevice(3);
    // Размер вектора - размер сетки в квадрате
    int vec_size = net_size * net_size;
    // Создание 2-х матриц, одна будет считаться на основе другой
    T *Anew = new T [vec_size],
      *A = new T [vec_size];

    // Инициализация матриц
    initialize_array(A, net_size);
    initialize_array(Anew, net_size);

    // Текущая ошибка
    T error = 0;
    // Счетчик итераций
    int iter;
    // Указатель для swap
    T *temp;
    //Память на девайсе
    double* A_dev, *Anew_dev, *error_dev, *A_err, *reduction_bufer = NULL;
	size_t reduction_bufer_size = 0;

	cudaError_t cudaStatus;
    // Выделение памяти на девайсе
    cudaStatus = cudaMalloc((void**)(&A_dev), sizeof(double) * vec_size); // Матрица
	cudaStatus = cudaMalloc((void**)(&Anew_dev), sizeof(double) * vec_size); // Еще одна матрица
	cudaMalloc((void**)&error_dev, sizeof(double)); // Ошибка (переменная)
	cudaStatus = cudaMalloc((void**)&A_err, sizeof(double) * vec_size); // Матрица ошибок
    // Скопировать заполненные массивы с хоста на девайс
    cudaStatus = cudaMemcpy(A_dev, A, sizeof(double) * vec_size, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(Anew_dev, Anew, sizeof(double) * vec_size, cudaMemcpyHostToDevice);
    // Здесь мы получаем размер временного буфера для редукции
    cub::DeviceReduce::Max(reduction_bufer, reduction_bufer_size, A_err, error_dev, vec_size);
    // Выделяем память для буфера
	cudaMalloc((void**)&reduction_bufer, reduction_bufer_size);

    // Флаг обновления ошибки на хосте для обработки условием цикла
    bool update_flag = true;
    for (iter = 0; iter < iter_max; ++iter)
    {
        // Сокращение количества обращений к CPU. Больше сетка - реже стоит сверять значения.
        //update_flag = !(iter % net_size);

        if (update_flag)
        {
            // зануление ошибки на GPU
            error = 0;
        }

        calculateMatrix<<<net_size - 1, net_size - 1>>>(A_dev, Anew_dev, net_size);

        // swap(A, Anew)
        temp = A, A = Anew, Anew = temp;
        // Проверить ошибку
        if (update_flag)
        {
            getErrorMatrix<<<net_size - 1, net_size - 1>>>(A_dev, Anew_dev, error_dev);
            // аналог reduction (max : error_dev) в OpenACC
            cub::DeviceReduce::Max(reduction_bufer, reduction_bufer_size, A_err, error_dev, vec_size);
            // Копировать ошибку с девайса на хост
            cudaMemcpy(&error, error_dev, sizeof(double), cudaMemcpyDeviceToHost);
            // Если ошибка не превышает точность, прекратить выполнение цикла
            if (error <= accuracy)
                break;
        }
    }

    std::cout.precision(2);
    if (res)
        print_array(A, net_size);
    std::cout << "iter=" << iter << ",\terror=" << error << std::endl;

	cudaFree(A);
	cudaFree(Anew);
	cudaFree(A_err);
	cudaFree(reduction_bufer);
	cudaFree(A_dev);
	cudaFree(Anew_dev);
}

int main(int argc, char *argv[])
{
    auto begin_main = std::chrono::steady_clock::now();
    int net_size = 128, iter_max = (int)1e6;
    T accuracy = 1e-6;
    bool res = false;
    for (int arg = 1; arg < argc; arg++)
    {
        std::string str = argv[arg];
        if (!str.compare("-res"))
            res = true;
        else
        {
            arg++;
            if (!str.compare("-a"))
                accuracy = STOD(argv[arg]);
            else if (!str.compare("-i"))
                iter_max = (int)std::stod(argv[arg]);
            else if (!str.compare("-s"))
                net_size = std::stoi(argv[arg]);
            else
            {
                std::cout << "Wrong args!";
                return -1;
            }
        }
    }
    calculate(net_size, iter_max, accuracy, res);
    auto end_main = std::chrono::steady_clock::now();
    int time_spent_main = std::chrono::duration_cast<std::chrono::milliseconds>(end_main - begin_main).count();
    std::cout << "The elapsed time is:\nmain\t\t\t" << time_spent_main << " ms\n";
    return 0;
}