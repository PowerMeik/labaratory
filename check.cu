#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>


//calculate error
__global__ void error(double * A1, double *anew1 , double * res, int n){

	     	int idx = blockIdx.x*blockDim.x + threadIdx.x; //find index
			if(idx<n*n){
						res[idx] = fabs(anew1[idx]-A1[idx]);
							}
}
//calculate new value of an array cell 
__global__ void compute(double * A1, double * anew1,int n){
		int j = blockIdx.x;// find index						          
			int i = threadIdx.x;
				if((j>0) && (j<n-1) && (i>0) && (i<n-1))
							anew1[i + j * n] = 0.25 * (A1[i + j * n - n] + A1[i + j * n + n] + A1[i + j * n - 1] + A1[i + j * n + 1]);
}

int main(int argc, char** argv) {
		//parametrs from command line
		int iter_max;
			int sizeofnet1;
				double accuracy;
					
					int blockSize;      // The launch configurator returned block size 
					    	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
						    	int gridSize;       // The actual grid size needed, based on input size
								
								if (argc<4){
											printf("not enough arguments");
													return 0;
														}
									accuracy = atof(argv[1]);
										sizeofnet1 = atoi(argv[2]);
											int n = sizeofnet1;
												iter_max = atoi(argv[3]);
													//check
													if (n>1024 || accuracy<0 || n<0 || iter_max<0){
																printf("wrong arguments");
																		return 0;
																			}
														
														//allocate memory for arrays
														double* A = (double*)calloc(n * n, sizeof(double));
															double* anew = (double*)calloc(n * n, sizeof(double));


																//calculate boundary conditions
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
																																		//allocate array with cuda function
																																		cudaMalloc((&err1), sizeof(double));
																																			cudaMalloc((&A1), sizeof(double) * (n * n));
																																				cudaMalloc((&anew1), sizeof(double) * (n*n));
																																					cudaMalloc((&tmp_arr), sizeof(double) *( n * n));
																																						//move arrays with boundary conditions to cuda arrays
																																						cudaMemcpy(A1, A, sizeof(double) * (n * n), cudaMemcpyHostToDevice);
																																							cudaMemcpy(anew1, anew, sizeof(double) * (n * n), cudaMemcpyHostToDevice);
												cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, error, 0, n*n);		
												gridSize = (n*n + blockSize - 1) / blockSize; 
																																					//calculate size of memory
										printf("%d\n %d\n", gridSize, blockSize);																																cub::DeviceReduce::Max(t_memory, t_memory_size, tmp_arr, err1, n*n);// t_memory = NULL, при первом вызове возвращает нужный размер, нужно выделить память, чтобы функция работала
																																										   	 cudaMalloc((&t_memory), t_memory_size);
																																											 	
																																											 	//main algorithm
																																											 	for (iter = 0; iter < iter_max && err>accuracy; iter++) {
																																															// n-1 - size of net and threads
																																															compute<<<gridSize, blockSize>>>(A1, anew1, n); // получать значения размера потоков и блоков во время выполнения программы (зависит от размера сетки)
																																																	//every 100 iterations calculate error
																																																	if (iter%100==0){
																																																					error<<<gridSize, blockSize>>>(A1, anew1, tmp_arr, n);
																																																								cub::DeviceReduce::Max(t_memory, t_memory_size, tmp_arr, err1, n*n);
																																																											//move answer to CPU
																																																											cudaMemcpy(&err, err1, sizeof(double), cudaMemcpyDeviceToHost);
																																																													}
																																																			//change arrays
																																																			double* tmp = A1;
																																																					A1 = anew1;
																																																							anew1 = tmp;
																																																								}
																																													//print answer
																																													printf("%d\n%lf", iter, err);
																																														//free memory
																																														cudaFree(err1);
																																															cudaFree(t_memory);
																																																cudaFree(A1);
																																																	cudaFree(anew1);
																																																		free(A);
																																																			free(anew);
																																																				return 0;
}
